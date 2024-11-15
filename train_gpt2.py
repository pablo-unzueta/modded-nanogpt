import os
import sys

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

with open(sys.argv[0]) as f:
    code = f.read()  # read the code of this file ASAP, for logging
import uuid
import glob
import time
from dataclasses import dataclass

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP
import inspect

# Original Code from Karpathy's NanoGPT
# https://github.com/karpathy/nanoGPT/blob/master/model.py


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class LayerNorm(nn.Module):
    "LayerNorm with optional bias."

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # k, q, v projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = False
        # self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        # if not self.flash:
        #     print(
        #         "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
        #     )
        #     # causal mask to ensure that attention is only applied to the left in the input sequence
        #     self.register_buffer(
        #         "bias",
        #         torch.tril(torch.ones(config.block_size, config.block_size)).view(
        #             1, 1, config.block_size, config.block_size
        #         ),
        #     )

    def forward(self, x):
        B, T, C = x.size()  # batch size, seq. length, and embedding dimensionality

        # calc q, k, v for all heads in batch and move head forward to the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:  # manual implementation of attention
            att = (q * k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        y = self.resid_dropout(self.c_proj(y))
        return y


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        print(f"Initializing GPT with vocab_size: {config.vocab_size}")
        print(f"Sequence length: {config.sequence_length}")

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.sequence_length, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )
        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()

        # Detailed validation
        max_token = torch.max(idx).item()
        min_token = torch.min(idx).item()
        print(f"Input tokens - shape: {idx.shape}, max: {max_token}, min: {min_token}")
        print(f"Embedding layer size: {self.transformer.wte.weight.shape}")

        if max_token >= self.config.vocab_size:
            raise ValueError(
                f"Token index {max_token} out of range (>= vocab_size {self.config.vocab_size})\n"
                f"Shape of idx: {idx.shape}\n"
                f"First few tokens: {idx[0, :10]}"
            )

        if min_token < 0:
            raise ValueError(f"Negative token index found: {min_token}")

        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos).unsqueeze(
            0
        )  # position embeddings of shape (t, n_embd)
        x = tok_emb + pos_emb
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
        )
        return loss.float()

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that don't require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any params that are 2D will be weight decayed, otherwise no.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if needed
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        return optimizer


@dataclass
class GPTConfig:
    vocab_size: int = 50304
    sequence_length: int = 1024
    n_layer: int = 12
    n_head: int = 6
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False


# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader


def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print(
            "---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README"
        )
        print(
            "---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try"
        )
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2]  # number of tokens (claimed)
    return ntok  # for now just return the number of tokens


def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2]  # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens


class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert (
            len(self.files) > 0
        ), f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total

        # kick things off
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self):  # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        try:
            buf = self.tokens[self.current_position : self.current_position + B * T + 1]
            print(f"Loading batch from position {self.current_position}")
            print(f"Buffer shape: {buf.shape}, max: {np.max(buf)}, min: {np.min(buf)}")

            if len(buf) < B * T + 1:
                print(
                    f"Warning: Buffer size {len(buf)} is less than expected {B * T + 1}"
                )

            buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
            x = (buf[:-1]).view(B, T)
            y = (buf[1:]).view(B, T)

            # Validation
            for name, tensor in [("x", x), ("y", y)]:
                max_val = tensor.max().item()
                if max_val >= num_vocab:
                    raise ValueError(
                        f"{name} contains token {max_val} >= vocab_size {num_vocab}\n"
                        f"Shape: {tensor.shape}\n"
                        f"Current file: {self.files[self.current_shard]}\n"
                        f"Position: {self.current_position}"
                    )

            self.current_position += B * T * self.num_processes
            if self.current_position + (B * T * self.num_processes + 1) > len(
                self.tokens
            ):
                self.advance()

            return x.cuda(), y.cuda()

        except Exception as e:
            print(f"Error in next_batch:")
            print(f"Current position: {self.current_position}")
            print(f"Current shard: {self.files[self.current_shard]}")
            print(f"Tokens shape: {self.tokens.shape}")
            raise e


# -----------------------------------------------------------------------------
# int main


@dataclass
class Hyperparameters:
    # data hyperparams
    input_bin: str = "data/fineweb10B/fineweb_train_*.bin"  # input .bin to train on
    input_val_bin: str = (
        "data/fineweb10B/fineweb_val_*.bin"  # input .bin to eval validation loss on
    )
    # optimization hyperparams
    batch_size: int = 8 * 64  # batch size, in sequences, across all devices
    device_batch_size: int = 32  # batch size, in sequences, per device
    sequence_length: int = 1024  # sequence length, in tokens
    num_iterations: int = 3242  # number of iterations to run
    warmup_iters: int = 0
    warmdown_iters: int = 926  # number of iterations of linear warmup/warmdown for triangular or trapezoidal schedule
    weight_decay: float = 0
    # evaluation and logging hyperparams
    val_loss_every: int = (
        125  # every how many steps to evaluate val loss? 0 for only at the end
    )
    val_tokens: int = 10485760  # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    save_every: int = (
        0  # every how many steps to save the checkpoint? 0 for only at the end
    )
    # adamw optimizer
    learning_rate = 6e-4  # max learning rate
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    decay_lr = True  # whether to decay the learning rate
    device: str = "cuda"


args = Hyperparameters()

# set up DDP (distributed data parallel). torchrun sets this env variable
assert torch.cuda.is_available()
dist.init_process_group(backend="nccl")
ddp_rank = int(os.environ["RANK"])
ddp_local_rank = int(os.environ["LOCAL_RANK"])
ddp_world_size = int(os.environ["WORLD_SIZE"])
device = f"cuda:{ddp_local_rank}"
torch.cuda.set_device(device)
print(f"using device: {device}")
master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.

# begin logging
logfile = None
if master_process:
    run_id = str(uuid.uuid4())
    logdir = "logs/%s/" % run_id
    os.makedirs(logdir, exist_ok=True)
    logfile = "logs/%s.txt" % run_id
    # create the log file
    with open(logfile, "w") as f:
        # begin the log by printing this file (the Python code)
        f.write("=" * 100 + "\n")
        f.write(code)
        f.write("=" * 100 + "\n")


def print0(s, logonly=False):
    if master_process:
        with open(logfile, "a") as f:
            if not logonly:
                print(s)
            f.write(s + "\n")


# log information about the hardware/software environment this is running on
# and print the full `nvidia-smi` to file
print0(
    f"Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\nnvidia-smi:"
)
import subprocess

result = subprocess.run(
    ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
)
print0(f"{result.stdout}", logonly=True)
print0("=" * 100, logonly=True)

# convenience variables
B, T = args.device_batch_size, args.sequence_length
# calculate the number of steps to take in the val loop.
assert args.val_tokens % (B * T * ddp_world_size) == 0
val_steps = args.val_tokens // (B * T * ddp_world_size)
# calculate the steps of gradient accumulation required to attain the desired global batch size.
assert args.batch_size % (B * ddp_world_size) == 0
train_accumulation_steps = args.batch_size // (B * ddp_world_size)

# load tokens
train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)
print0(
    f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files"
)
print0(
    f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files"
)
print0("=" * 100, logonly=True)
x, y = train_loader.next_batch()

# there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency. suggested to me by @Grad62304977.
# this originates from Karpathy's experiments.
num_vocab = 50304
config = GPTConfig()
print(f"Creating model with config:")
print(f"vocab_size: {config.vocab_size}")
print(f"sequence_length: {config.sequence_length}")
print(f"n_layer: {config.n_layer}")
print(f"n_head: {config.n_head}")
print(f"n_embd: {config.n_embd}")
model = GPT(config)
# model = model.to(args.device).bfloat16()

# here we wrap model into DDP container
model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module  # always contains the "raw" unwrapped model
# optimizer
optimizer = raw_model.configure_optimizers(
    args.weight_decay, args.learning_rate, (args.beta1, args.beta2), args.device
)
checkpoint = None  # free up memory
# CUDNN attention is ~4ms faster than Flash, but doesn't get selected by default in PyTorch 2.5.1
# from torch.backends.cuda import (
#     enable_cudnn_sdp,
#     enable_flash_sdp,
#     enable_math_sdp,
#     enable_mem_efficient_sdp,
# )

# enable_cudnn_sdp(True)
# enable_flash_sdp(False)
# enable_mem_efficient_sdp(False)
# enable_math_sdp(False)

# init the optimizer(s)
# optimizer1 = torch.optim.Adam(
#     [raw_model.transformer.wte.weight], lr=0.3, betas=(0.9, 0.95), fused=True
# )
# optimizer2 = torch.optim.Adam(
#     [raw_model.lm_head.weight], lr=0.002, betas=(0.9, 0.95), fused=True
# )
# params = list(raw_model.transformer.h.parameters())
# matrix_params = [p for p in params if p.ndim == 2]
# scalar_params = [p for p in params if p.ndim < 2]
# optimizer3 = torch.optim.Adam(matrix_params, lr=0.001, betas=(0.9, 0.95), fused=True)
# # optimizer3 = Muon(matrix_params, lr=0.02, momentum=0.95)
# optimizer4 = torch.optim.Adam(
#     scalar_params, lr=0.02, betas=(0.9, 0.95), fused=True
# )  # note that this learning rate is neither sensitive nor tuned
# optimizers = [optimizer1, optimizer2, optimizer3, optimizer4]


# learning rate decay scheduler (linear warmup and warmdown)
def get_lr(it):
    assert it <= args.num_iterations
    # 1) linear warmup for warmup_iters steps
    if it < args.warmup_iters:
        return (it + 1) / args.warmup_iters
    # 2) constant lr for a while
    elif it < args.num_iterations - args.warmdown_iters:
        return 1.0
    # 3) linear warmdown
    else:
        decay_ratio = (args.num_iterations - it) / args.warmdown_iters
        return decay_ratio


schedulers = [torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)]

# Start training loop
training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.time()
# begin training
train_loader.reset()
for step in range(args.num_iterations + 1):
    last_step = step == args.num_iterations
    # This effectively ignores timing first 10 steps, which are slower for weird reasons.
    # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
    # steps with dummy data first, and then re-initialize the model and reset the loader.
    if step == 10:
        training_time_ms = 0
        t0 = time.time()
    timed_steps = (
        float("nan") if step <= 11 else (step - 10) + 1
    )  # <= 11 to avoid bug in val

    # once in a while evaluate the validation dataset
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        # run validation batches
        model.eval()
        val_loader.reset()
        val_loss = 0.0
        for _ in range(val_steps):
            with torch.no_grad():
                x_val, y_val = val_loader.next_batch()
                val_loss += model(x_val, y_val)
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss /= val_steps
        # log val loss to console and to logfile
        print0(
            f"step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms"
        )
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

    if master_process and (
        last_step or (args.save_every > 0 and step % args.save_every == 0)
    ):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        # save the state of the training process
        log = dict(
            step=step,
            code=code,
            model=raw_model.state_dict(),
            optimizer=optimizer.state_dict(),
        )
        torch.save(log, "logs/%s/state_step%06d.pt" % (run_id, step))
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

    # bit confusing: we want to make sure to eval on 0th iteration
    # but also after the very last iteration. so we loop for step <= num_iterations
    # instead of just < num_iterations (one extra due to <=), only to do
    # the validation/sampling one last time, and then we break right here as we're done.
    if last_step:
        break

    # --------------- TRAINING SECTION BEGIN -----------------
    model.train()
    for i in range(1, train_accumulation_steps + 1):
        # forward pass
        loss = model(x, y)
        train_loss = loss.detach()
        # advance the dataset for the next batch
        x, y = train_loader.next_batch()
        # backward pass
        if i < train_accumulation_steps:
            with model.no_sync():  # there's no need to sync gradients every accumulation step
                loss.backward()
        else:
            loss.backward()  # just sync on the last step
    for p in model.parameters():
        p.grad /= train_accumulation_steps
    # step the optimizers and schedulers
    optimizer.step()
    schedulers.step()
    # null the gradients
    model.zero_grad(set_to_none=True)

    # --------------- TRAINING SECTION END -------------------
    # everything that follows now is just diagnostics, prints, logging, etc.

    # dist.all_reduce(train_loss, op=dist.ReduceOp.AVG) # all-reducing the training loss would be more correct in terms of logging, but slower
    approx_time = training_time_ms + 1000 * (time.time() - t0)
    print0(
        f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms"
    )

if master_process:
    print(
        f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB"
    )

# -------------------------------------------------------------------------
# clean up nice
dist.destroy_process_group()
