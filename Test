import torch
from torch import nn
from torchsummary import summary
import time

# ---------------- DEVICE ---------------- #
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsing Device: {device}\n")

# ---------------- INITIALIZE MODELS ---------------- #
gen = Generator().to(device)
disc = Discriminator().to(device)

# ---------------- MODEL SUMMARY ---------------- #
print("=" * 60)
print("GENERATOR SUMMARY")
print("=" * 60)
summary(gen, (3, 24, 24))

print("\n" + "=" * 60)
print("DISCRIMINATOR SUMMARY")
print("=" * 60)
summary(disc, (3, 96, 96))

# ---------------- PARAMETER COUNT ---------------- #
gen_params = sum(p.numel() for p in gen.parameters())
disc_params = sum(p.numel() for p in disc.parameters())

print("\n" + "=" * 60)
print(f"Generator Parameters     : {gen_params:,}")
print(f"Discriminator Parameters : {disc_params:,}")
print("=" * 60)

# ---------------- RANDOM INPUT TEST ---------------- #
print("\nRunning Random Input Test...\n")

batch_size = 4
lr_size = 24

low_res = torch.randn((batch_size, 3, lr_size, lr_size)).to(device)

start = time.time()

with torch.no_grad():
    sr_output = gen(low_res)
    disc_output = disc(sr_output)

end = time.time()

# ---------------- SHAPE CHECKS ---------------- #
print("=" * 60)
print("OUTPUT SHAPES")
print("=" * 60)

print(f"Low Resolution Input Shape  : {low_res.shape}")
print(f"Super Resolution Shape      : {sr_output.shape}")
print(f"Discriminator Output Shape  : {disc_output.shape}")

# ---------------- VALUE CHECKS ---------------- #
print("\n" + "=" * 60)
print("OUTPUT VALUE CHECKS")
print("=" * 60)

print(f"SR Min Value  : {sr_output.min().item():.4f}")
print(f"SR Max Value  : {sr_output.max().item():.4f}")
print(f"SR Mean Value : {sr_output.mean().item():.4f}")

print(f"\nDiscriminator Output:")
print(disc_output)

# ---------------- NAN CHECK ---------------- #
print("\n" + "=" * 60)
print("NaN CHECK")
print("=" * 60)

if torch.isnan(sr_output).any():
    print("NaN values found in Generator Output")
else:
    print("No NaN values in Generator Output")

if torch.isnan(disc_output).any():
    print("NaN values found in Discriminator Output")
else:
    print("No NaN values in Discriminator Output")

# ---------------- FORWARD PASS SPEED ---------------- #
print("\n" + "=" * 60)
print("PERFORMANCE TEST")
print("=" * 60)

print(f"Forward Pass Time: {(end-start):.4f} seconds")

# ---------------- MULTIPLE SIZE TEST ---------------- #
print("\n" + "=" * 60)
print("MULTI-INPUT SIZE TEST")
print("=" * 60)

sizes = [16, 24, 32, 48]

for size in sizes:
    try:
        x = torch.randn((1, 3, size, size)).to(device)

        with torch.no_grad():
            out = gen(x)

        print(f"Input {size}x{size} --> Output {out.shape[2]}x{out.shape[3]}  SUCCESS")

    except Exception as e:
        print(f"Input {size}x{size} FAILED")
        print(e)

# ---------------- BACKPROP TEST ---------------- #
print("\n" + "=" * 60)
print("BACKPROPAGATION TEST")
print("=" * 60)

criterion = nn.MSELoss()

fake_hr = torch.randn_like(sr_output).to(device)

optimizer = torch.optim.Adam(gen.parameters(), lr=1e-4)

optimizer.zero_grad()

generated = gen(low_res)

loss = criterion(generated, fake_hr)

loss.backward()

optimizer.step()

print(f"Backpropagation Successful")
print(f"Loss Value: {loss.item():.6f}")

# ---------------- TRAINING LOOP TEST ---------------- #
print("\n" + "=" * 60)
print("MINI TRAINING LOOP TEST")
print("=" * 60)

g_optimizer = torch.optim.Adam(gen.parameters(), lr=1e-4)
d_optimizer = torch.optim.Adam(disc.parameters(), lr=1e-4)

bce = nn.BCEWithLogitsLoss()

for step in range(3):

    low_res = torch.randn((2, 3, 24, 24)).to(device)
    high_res = torch.randn((2, 3, 96, 96)).to(device)

    # ----- Train Discriminator ----- #
    fake = gen(low_res)

    real_out = disc(high_res)
    fake_out = disc(fake.detach())

    real_loss = bce(real_out, torch.ones_like(real_out))
    fake_loss = bce(fake_out, torch.zeros_like(fake_out))

    d_loss = real_loss + fake_loss

    d_optimizer.zero_grad()
    d_loss.backward()
    d_optimizer.step()

    # ----- Train Generator ----- #
    fake_out = disc(fake)

    g_loss = bce(fake_out, torch.ones_like(fake_out))

    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()

    print(
        f"Step [{step+1}/3] "
        f"D Loss: {d_loss.item():.4f} | "
        f"G Loss: {g_loss.item():.4f}"
    )

print("\n" + "=" * 60)
print("ALL TESTS COMPLETED SUCCESSFULLY")
print("=" * 60)
