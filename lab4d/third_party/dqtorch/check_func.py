import time
import torch


def check_close(name, out1, out2, rtol=None, atol=None, quiet=True):
    if rtol is None:
        rtol = 1e-3 if out1.dtype == torch.float32 else 1e-8
    if atol is None:
        atol = 5e-5 if out2.dtype == torch.float32 else 5e-15

    passed = torch.allclose(out1, out2, rtol=rtol, atol=atol)
    if passed:
        if not quiet:
            print(f"{name}: Passed")
    else:
        is_wrong = torch.logical_not(torch.isclose(out1, out2, rtol=rtol, atol=atol))
        idxs_wrong = torch.nonzero(is_wrong, as_tuple=True)
        num_wrong = torch.sum(is_wrong)
        num_total = torch.numel(is_wrong)
        print(f"{name}:\tPercent Error {num_wrong} / {num_total} = {num_wrong / num_total}")
        if not quiet:
            print("idxs")
            print(idxs_wrong)
            print("out1")
            print(out1[idxs_wrong])
            print("out2")
            print(out2[idxs_wrong])

    return passed


def check_func(name, args, func_custom, func_ref, n_trials=100):
    """Check `func_custom` outputs and first-order gradients against `func_ref`

    Args:
        name (str): Name of this test
        func_custom: Custom function to test
        func_ref: Reference function to test
        args (torch.Tensor or Tuple(torch.Tensor)): Tuple of args to both functions
        n_trials (int): Number of trials to average timings over
        quiet (bool): If True, suppresses print output
    Returns:
        fw_custom (float): Time to compute custom forward pass
        bw_custom (float): Time to compute custom backward pass
        fw_ref (float): Time to compute custom forward pass
        bw_ref (float): Time to compute custom backward pass
    """
    if not isinstance(args, tuple):
        args = (args,)
    
    fw_custom = 0
    bw_custom = 0
    fw_ref = 0
    bw_ref = 0
    passed = True
    for it in range(n_trials):
        args1 = tuple(
            tuple(xx.clone().requires_grad_(True) for xx in x)
            if isinstance(x, tuple)
            else x.clone().requires_grad_(True)
            for x in args
        )
        torch.cuda.synchronize()
        t1 = time.time()
        out1 = func_custom(*args1)
        torch.cuda.synchronize()
        t2 = time.time()
        fw_custom += t2 - t1
        if isinstance(out1, tuple):
            out1 = torch.cat(out1, dim=-1)
        torch.cuda.synchronize()
        t1 = time.time()
        out1.sum().backward()
        torch.cuda.synchronize()
        t2 = time.time()
        bw_custom += t2 - t1
        
        args2 = tuple(
            tuple(xx.clone().requires_grad_(True) for xx in x)
            if isinstance(x, tuple)
            else x.clone().requires_grad_(True)
            for x in args
        )
        torch.cuda.synchronize()
        t1 = time.time()
        out2 = func_ref(*args2)
        torch.cuda.synchronize()
        t2 = time.time()
        fw_ref += t2 - t1
        if isinstance(out2, tuple):
            out2 = torch.cat(out2, dim=-1)
        torch.cuda.synchronize()
        t1 = time.time()
        out2.sum().backward()
        torch.cuda.synchronize()
        t2 = time.time()
        bw_ref += t2 - t1

        passed = passed and check_close(f"{name}, out", out1, out2)
        for i, (arg1, arg2) in enumerate(zip(args1, args2)):
            if isinstance(arg1, tuple) and isinstance(arg2, tuple):
                for j, (arg1_elt, arg2_elt) in enumerate(zip(arg1, arg2)):
                    passed = passed and check_close(f"{name}, arg {i}.{j} grad", arg1_elt.grad, arg2_elt.grad)
            else:
                passed = passed and check_close(f"{name}, arg {i} grad", arg1.grad, arg2.grad)

    fw_custom /= n_trials
    bw_custom /= n_trials
    fw_ref /= n_trials
    bw_ref /= n_trials

    if passed:
        print(
            f"{name} passed:\n"
            f"  fw {fw_ref / fw_custom:7.3f}x\tref {1e6 * fw_ref:9.3f} us\tcustom {1e6 * fw_custom:9.3f} us\n"
            f"  bw {bw_ref / bw_custom:7.3f}x\tref {1e6 * bw_ref:9.3f} us\tcustom {1e6 * bw_custom:9.3f} us"
        )
    else:
        print(f"{name} failed")

    return passed
