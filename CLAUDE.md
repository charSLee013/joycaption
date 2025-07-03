
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## High-Level Architecture

JoyCaption is an image captioning Visual Language Model (VLM) built upon the Llama 3.1 model and HuggingFace transformers. Its primary function is to generate diverse image captions based on user-defined prompts and styles.

The codebase is structured into the following main components:
-   **Core Model Logic**: Handled by Python scripts that utilize the `transformers` library for loading and interacting with the JoyCaption VLM.
-   **`finetuning/`**: Contains scripts and documentation for fine-tuning the JoyCaption model on custom datasets. This allows for adapting the model's captioning style or domain. The `train.py` script is central to this process.
-   **`gradio-app/`**: Hosts a Gradio-based web application for easy interaction with the JoyCaption model. It provides a user interface for single image captioning and batch processing with various caption types and model quantization options.
-   **`scripts/`**: Contains utility scripts, such as `dual_caption.py` for generating dual annotations.

The model supports various captioning modes, including descriptive, straightforward, Stable Diffusion-style prompts, MidJourney-style prompts, and different Booru tag formats (Danbooru, e621, Rule34, generic Booru-like tags), as well as art critic analysis, product listings, and social media posts.

## Common Development Tasks

-   **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
-   **Run Gradio Web Application**:
    ```bash
    cd gradio-app
    python app.py
    ```
-   **Run vLLM Server for Inference**:
    ```bash
    vllm serve fancyfeast/llama-joycaption-beta-one-hf-llava --max-model-len 4096 --enable-prefix-caching
    ```
    For Windows with Docker:
    ```bash
    docker run --gpus all --ipc=host -p 8000:8000 -v "%USERPROFILE%\.cache\huggingface:/root/.cache/huggingface" vllm/vllm-openai:latest --model fancyfeast/llama-joycaption-beta-one-hf-llava --max-model-len 4096 --enable-prefix-caching
    ```
-   **Fine-tuning JoyCaption**:
    Refer to `finetuning/README.md` for detailed instructions. An example training command is:
    ```bash
    torchrun --standalone --nproc_per_node=1 train.py --wandb-project finetune-2 --device-batch-size 4 --dataset ../instruction-dataset/answers-train.json --max-samples 1800 --images-path ../instruction-dataset --test-every 2000 --test-size 128
    ```
    To use a finetuned model with vLLM, you need to merge the LORA weights:
    ```python
    model = model.merge_and_unload(progressbar=True)
    model.save_pretrained("./questions-cuu2y0sx")
    processor.save_pretrained("./questions-cuu2y0sx")
    ```
    Then serve with vLLM:
    ```bash
    vllm serve ./questions-cuu2y0sx --max-model-len 4096 --enable-prefix-caching
    ```

**Note**: There are no explicit linting or testing commands provided in the existing documentation. If required, these should be identified or established based on the project's development practices.

## 测试规范详细要求
1. 测试必须覆盖所有实现的功能点
2. 测试文件必须包含：
   - 单元测试
   - 集成测试（如适用）
   - 边界条件测试
3. 测试必须真实可执行，不能仅为示例代码
4. 测试通过标准：运行无错误、断言全部通过
5. 测试报告必须包含覆盖率信息
6. 对于每个测试，提供预期结果和实际结果的对比

## 完成提示
当任务全部完成并通过测试验证后，say "搞完了"
