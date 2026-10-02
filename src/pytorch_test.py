import torch

# Use double precision like your C++ code
torch.set_default_dtype(torch.float64)

# Parameter to optimize
x = torch.nn.Parameter(torch.tensor([20.0]))

# Optimizer
optimizer = torch.optim.LBFGS([x], lr=1.0, max_iter=100, line_search_fn="strong_wolfe")

iteration = 0


def closure():
    global iteration

    optimizer.zero_grad()

    # f(x) = (x-7)^2 - 30
    loss = (x - 7.0) ** 2 - 30

    loss.backward()

    print(
        f"iter={iteration:3d} "
        f"x={x.item():.8f} "
        f"f(x)={loss.item():.8f} "
        f"grad={x.grad.item():.8f}"
    )

    iteration += 1

    return loss


optimizer.step(closure)

print("\nFinal result")
print("x =", x.item())
print("f(x) =", ((x - 7.0) ** 2 - 30).item())
