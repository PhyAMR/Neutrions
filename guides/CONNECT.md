Here is a complete, production-ready `README.md` guide designed for you. It covers both the simple setup you just did and explains the best way to scale up as you add more users down the line.

You can save this text directly as a `README.md` file in your project folder or your home directory.

---

```markdown
# CloudVeneto VM Team Onboarding & SSH Access Guide

This guide details how to add multiple users to our private cloud virtual machine (`10.67.22.85`) routed securely via the entry gateway (`gate.cloudveneto.it`).

---

## 1. System Infrastructure Overview

Because our destination VM sits on a isolated private network tier, direct incoming public connections are blocked. All team members must jump through the gateway hop to reach the system.

*   **Gateway Node (First Hop):** `gate.cloudveneto.it`
*   **Destination VM Node:** `10.67.22.85` (Ubuntu Linux)

---

## 2. Server Configuration Setup (Run Once)

To ensure the VM allows key-based or password authentication smoothly without background cloud configurations overwriting our changes, ensure the following keys are active on the VM.

Log into your VM and run:
```bash
sudo nano /etc/ssh/sshd_config.d/60-cloudimg-settings.conf

```

Verify/update these lines to ensure they match:

```text
PasswordAuthentication yes
KbdInteractiveAuthentication yes

```

Save (`Ctrl+O`, `Enter`) and exit (`Ctrl+X`), then reload the service daemon:

```bash
sudo systemctl restart ssh

```

---

## 3. How to Add a New User to the VM

Follow these steps exactly every time a new team member requires access to the cluster. Replace `USERNAME` with their short account handle (e.g., `chitaban`) and `'FULL_NAME'` with their actual name.

### Step 3.1: Provision the Local OS Account

Log into the VM as an administrator (`ubuntu`) and run:

```bash
# Create user, assign home directory, and set default bash shell environment
sudo useradd -m -c 'FULL_NAME' -s /bin/bash USERNAME

```

### Step 3.2: Option A - Password-Based Onboarding (Quick Start)

If the user does not have an SSH key yet and needs immediate access:

```bash
# 1. Set a compliant temporary password (avoid matching the username string)
sudo passwd USERNAME

# 2. Force reset password aging rules so they change it on initial hand-shake
sudo chage -d 0 USERNAME

# 3. Clean any reactive brute-force lock rules
sudo usermod -U USERNAME

```

### Step 3.3: Option B - SSH Key-Based Onboarding (Highly Recommended)

To bypass password prompts entirely and maximize infrastructure security, collect the user's public key (`~/.ssh/id_rsa.pub`) and drop it into their home folder space on the VM:

```bash
# 1. Initialize safe system directories for the user profile
sudo mkdir -p /home/USERNAME/.ssh

# 2. Append their raw text key string into the authorized inventory target
sudo nano /home/USERNAME/.ssh/authorized_keys
# [Paste their public key string here, save and exit]

# 3. CRUCIAL: Fix ownership and lock file permissions so SSH allows the handshake
sudo chown -R USERNAME:USERNAME /home/USERNAME/.ssh
sudo chmod 700 /home/USERNAME/.ssh
sudo chmod 600 /home/USERNAME/.ssh/authorized_keys

```

---

## 4. Instructions to Send to the New User

Copy and paste the following markdown template block and send it to your new team member so they can set up their computer.

---

### 📥 Welcome to the Team Server! Here is your SSH Configuration Hook

To access the cloud cluster environment, you must map out the jump route on your local laptop.

#### Step A: Configure your local environment

Open a terminal on your personal machine, create/open your local config file:

```bash
nano ~/.ssh/config

```

Paste the following blocks inside (replace `YOUR_GATEWAY_USER` with the gateway credentials you were assigned, and `YOUR_VM_USER` with your newly minted user name):

```text
# Step 1: Entry Gateway Access Hub
Host gate
    HostName gate.cloudveneto.it
    User YOUR_GATEWAY_USER

# Step 2: Destination Compute Target 
Host cloud-vm
    HostName 10.67.22.85
    User YOUR_VM_USER
    ProxyJump gate

```

#### Step B: Establish Connection

Run the connection tool command string below:

```bash
ssh cloud-vm

```

* **If using password auth:** You will be prompted to enter your temporary credentials. The system will force you to change your password immediately.
* **If using key-based auth:** The system will effortlessly open your remote session prompt (`USERNAME@cloud-vm:~$`) without any password interruptions!

---

```

```