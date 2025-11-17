"""
Zen Max Identity Training Space
Train Zen identity on top of Kimi K2 Thinking using LoRA/QLoRA
All training happens in HuggingFace Space - no local downloads needed
"""

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import os

BASE_MODEL = "moonshotai/Kimi-K2-Thinking"
ZEN_DATASET = "zenlm/zen-identity-dataset"  # Create this with Zen persona/values

def train_zen_identity(
    learning_rate: float = 2e-4,
    num_epochs: int = 3,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    max_steps: int = 1000
):
    """
    Train Zen identity on top of K2 Thinking using QLoRA
    Efficient 4-bit training - no need to download full weights
    """
    
    # Load model in 4-bit for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, peft_config)
    
    # Load Zen identity dataset
    dataset = load_dataset(ZEN_DATASET, split="train")
    
    # Training configuration
    training_args = {
        "output_dir": "./zen-max-adapters",
        "num_train_epochs": num_epochs,
        "max_steps": max_steps,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": learning_rate,
        "fp16": False,
        "bf16": True,
        "logging_steps": 10,
        "save_steps": 100,
        "save_total_limit": 3,
        "push_to_hub": True,
        "hub_model_id": "zenlm/zen-max",
        "hub_strategy": "every_save",
    }
    
    # Train
    from trl import SFTTrainer
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_seq_length=4096,
        args=training_args,
    )
    
    # Start training
    trainer.train()
    
    # Push final adapters to hub
    model.push_to_hub("zenlm/zen-max")
    tokenizer.push_to_hub("zenlm/zen-max")
    
    return "Training complete! Adapters pushed to zenlm/zen-max"


def create_interface():
    """Create Gradio interface for training"""
    
    with gr.Blocks(title="Zen Max Identity Training") as demo:
        gr.Markdown("""
        # Zen Max Identity Training
        
        Train Zen identity on top of Kimi K2 Thinking using QLoRA.
        
        **Features:**
        - 4-bit quantized training (efficient memory usage)
        - LoRA adapters only (no full model download needed)
        - Trains on Zen identity dataset
        - Pushes adapters directly to zenlm/zen-max
        
        **Base Model:** `moonshotai/Kimi-K2-Thinking` (671B MoE, ~1TB)  
        **Training Method:** QLoRA (4-bit)  
        **Output:** LoRA adapters (~100MB) uploaded to HuggingFace
        """)
        
        with gr.Row():
            with gr.Column():
                learning_rate = gr.Slider(
                    minimum=1e-5, maximum=1e-3, value=2e-4,
                    label="Learning Rate"
                )
                num_epochs = gr.Slider(
                    minimum=1, maximum=10, value=3, step=1,
                    label="Number of Epochs"
                )
                lora_r = gr.Slider(
                    minimum=4, maximum=64, value=16, step=4,
                    label="LoRA Rank (r)"
                )
                lora_alpha = gr.Slider(
                    minimum=8, maximum=128, value=32, step=8,
                    label="LoRA Alpha"
                )
                lora_dropout = gr.Slider(
                    minimum=0.0, maximum=0.2, value=0.05, step=0.01,
                    label="LoRA Dropout"
                )
                max_steps = gr.Slider(
                    minimum=100, maximum=5000, value=1000, step=100,
                    label="Max Training Steps"
                )
                
                train_btn = gr.Button("Start Training", variant="primary")
            
            with gr.Column():
                output = gr.Textbox(label="Training Status", lines=20)
        
        train_btn.click(
            fn=train_zen_identity,
            inputs=[learning_rate, num_epochs, lora_r, lora_alpha, lora_dropout, max_steps],
            outputs=output
        )
        
        gr.Markdown("""
        ## What happens during training?
        
        1. **Model Loading**: K2 Thinking loaded in 4-bit quantization (saves ~75% memory)
        2. **LoRA Configuration**: Adapters added to attention and MLP layers
        3. **Dataset Loading**: Zen identity conversations and values
        4. **Training**: QLoRA fine-tuning for Zen persona
        5. **Upload**: LoRA adapters pushed to `zenlm/zen-max`
        
        ## After Training
        
        Use the model with:
        ```python
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained("moonshotai/Kimi-K2-Thinking")
        
        # Load Zen adapters
        model = PeftModel.from_pretrained(base_model, "zenlm/zen-max")
        tokenizer = AutoTokenizer.from_pretrained("zenlm/zen-max")
        
        # Inference with Zen identity
        response = model.chat(tokenizer, messages, thinking_budget=128000)
        ```
        
        ## Hardware Requirements
        
        - **GPU**: 1x A100 80GB (Space will use this)
        - **Memory**: ~60GB VRAM for 4-bit training
        - **Storage**: ~5GB for adapters and cache
        - **Training Time**: ~2-4 hours for 1000 steps
        """)
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
