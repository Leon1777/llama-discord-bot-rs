## Rust + [Llama.cpp.rs](https://docs.rs/llama-cpp-2/latest/llama_cpp_2/) + [Serenity](https://docs.rs/serenity/latest/serenity/) (WIP)

### Run the Bot

1. **Install Rust**  
   Download and install Rust from [https://rustup.rs](https://rustup.rs).

2. **Clone and Build**

   ```bash
   git clone https://github.com/Leon1777/llama-discord-bot-rs.git
   cd your-repo-name
   cargo build --release
   ```

3. **Configure**  
   Add your **Discord Token** to a `.env` file:

   ```env
   DISCORD_TOKEN=your_token
   ```

4. **Run the Bot**
   ```bash
   cargo run --release
   ```

---

### Commands

- **`!ask <question>`**: Ask the bot anything.

---

### Features

- Stores the last 5 messages to maintain context (resets on restart)

---

### Models Tested

- [Mistral-Nemo-Instruct-2407](https://huggingface.co/bartowski/Mistral-Nemo-Instruct-2407-GGUF)
- [Meta-Llama 3.1-8B-Instruct](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF)

---

### Convert HuggingFace Model to GGUF Format

1. **Download the Model**

   ```python
   from huggingface_hub import snapshot_download

   model_id = "repo/model"
   snapshot_download(repo_id=model_id, local_dir="model_name",
                     local_dir_use_symlinks=False, revision="main")
   ```

2. **Set Up Llama.cpp**

   ```bash
   git clone https://github.com/ggerganov/llama.cpp.git
   cd llama.cpp
   pip install -r requirements.txt
   ```

3. **Convert to GGUF**
   ```bash
   python convert_hf_to_gguf model_folder --outfile model_name.gguf --outtype f16
   ```

---

### Quantize FP16

1. **Set Up Llama.cpp for Quantization**

   ```bash
   git clone https://github.com/ggerganov/llama.cpp.git
   cd llama.cpp
   mkdir build
   cd build
   cmake .. -G "Ninja" -DCMAKE_BUILD_TYPE=Release
   cmake --build .
   cd bin
   ```

2. **Quantize the Model**
   ```bash
   ./llama-quantize 3.1-8B.fp16.gguf 3.1-8B.q6_K.gguf Q6_K
   ```

---

### Example: Download a Model from HuggingFace

```python
from huggingface_hub import snapshot_download

model_id = "mistralai/Mistral-Large-Instruct-2411"
snapshot_download(
    repo_id=model_id,
    local_dir="models/Mistral-Large-Instruct-2411",
    local_dir_use_symlinks=False,
    revision="main",
)
```

---

This bot is lightweight, fast, and highly configurable, thanks to **Rust** **Llama.cpp.rs**, and **Serenity**.
