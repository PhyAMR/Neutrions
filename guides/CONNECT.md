# Connecting to your VM

Each of us works on **their own virtual machine** on CloudVeneto, and we share code through GitHub (see [GIT_WORKFLOW.md](GIT_WORKFLOW.md)). This guide shows how to create your VM and log in to it from your laptop.

## How the connection works

VMs on CloudVeneto have no public address, so you reach yours in two hops. Once SSH is configured (step 3), it makes both hops for you.

```text
your laptop ──ssh──▶ gate.cloudveneto.it ──ssh──▶ your VM (10.67.x.x, Ubuntu)
                     (gateway)
```

You need:

- **Your CloudVeneto account:** it gives you the dashboard and the gateway login.
- **Your VM's IP address:** you get it when you create the VM (step 2).

## 1. Create an SSH key (once per laptop)

Skip this step if `~/.ssh/id_ed25519.pub` already exists.

```bash
ssh-keygen -t ed25519 -C "your.email@example.com"
```

Press Enter to accept the default location. A passphrase is optional but recommended.

> The file ending in **`.pub`** is your public key, and you can share it. The file without `.pub` is your private key. It never leaves your laptop, not even to the gateway.

## 2. Create your VM

In the CloudVeneto dashboard:

1. Under **Key Pairs**, choose **Import Public Key** and paste the output of `cat ~/.ssh/id_ed25519.pub`. If the dashboard refuses an `ed25519` key, make an RSA one instead (`ssh-keygen -t rsa -b 4096`) and import `~/.ssh/id_rsa.pub`.
2. Under **Instances**, launch a new instance with an **Ubuntu** image and select your key pair.
3. Once it is running, write down its IP address (`10.67.x.x`).

The default user on Ubuntu images is `ubuntu`. Your key is already installed for that user, so there's nothing else to set up on the VM.

## 3. Configure SSH (once per laptop)

Open `~/.ssh/config` (create it if it doesn't exist; on Windows it is `C:\Users\<you>\.ssh\config`) and add:

```text
Host gate
    HostName gate.cloudveneto.it
    User YOUR_CLOUDVENETO_USER

Host cloud-vm
    HostName YOUR_VM_IP
    User ubuntu
    ProxyJump gate
```

Replace `YOUR_CLOUDVENETO_USER` and `YOUR_VM_IP`.

`ProxyJump` passes your key through the gateway without copying it there. If an older guide tells you to copy your private key to the gateway, don't: you don't need to.

## 4. Connect

```bash
ssh cloud-vm
```

- You are asked for your **gateway password** first. After that, your key logs you in to the VM.
- The first time, SSH asks you to confirm each machine's fingerprint. Type `yes`.

**VS Code:** install the *Remote - SSH* extension, run **Remote-SSH: Connect to Host…** and choose `cloud-vm`. It uses the same config file.

**Copying files** from your laptop to the VM uses the same config, for example `scp myfile.txt cloud-vm:~/`.

Once you are in, follow [GIT_WORKFLOW.md](GIT_WORKFLOW.md) to set up git and get the code.

### If it doesn't work

| Message | What to check |
|---|---|
| `Could not resolve hostname gate` | The `Host gate` block is missing or misspelled in `~/.ssh/config`. |
| `Permission denied` before you reach the VM | Your gateway username or password. Test the first hop on its own with `ssh gate`. |
| `Permission denied (publickey)` at the VM | The VM was launched without your key pair, or `User` isn't `ubuntu`. |
| `Connection timed out` at the VM | Is the VM running in the dashboard? Is `YOUR_VM_IP` right? |
| Anything else | Run `ssh -v cloud-vm` and look at the last lines, or send them to a teammate. |
