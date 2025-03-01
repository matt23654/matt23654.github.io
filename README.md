# Enhancing Reasoning in Distilled Language Models with GRPO: A DeepSeek R1 Distill-Qwen-1.5B Case Study

## Introduction

Since the release of DeepSeek-R1, there has been huge community interest in applying GRPO to reasoning models. However, large models like DeepSeek-R1 are very computationally expensive to train from scratch and so the open-source community has exploded with large numbers of proof of concepts and examples based on small models. [TinyZero](https://github.com/Jiayi-Pan/TinyZero) and [Mini-R1](https://www.philschmid.de/mini-deepseek-r1) are examples showing that small 3b models can learn complex reasoning directly from scratch using GRPO.

It has also been shown that reasoning capabilities can be learned for smaller models by using filtered traces of larger models [S1](https://huggingface.co/simplescaling/s1.1-32B). This appears to have several advantages, firstly it is way cheaper than training a model like DeepSeek-R1 from scratch and secondly the smaller model is then cheaper to inference and can be used on far cheaper hardware than DeepSeek-R1 which requires an optimized datacentre deployment.

Unfortunately, distilled models suffer from some drawbacks. They do not seem to retain all the capabilities of their teacher models and have gaps in their capabilities. It has been proposed originally by [DeepSeek](https://arxiv.org/abs/2501.12948) that taking a reasoning model as a base and then applying further GRPO to it could yield good results.

In this article, I will share my experience of using GRPO to enhance the reasoning capabilities of **DeepSeek-R1-Distill-Qwen-1.5B** on a toy problem to demonstrate it is not just possible to enhance existing performance [DeepScaleR](https://github.com/agentica-project/deepscaler) but that it is possible to teach it new reasoning domains as well, to some extent.

## Toy Reasoning

The countdown problem, popularized by [TinyZero](https://github.com/Jiayi-Pan/TinyZero), is an excellent toy problem to study. Here's why:

- **Not Too Difficult**: It strikes a balance between being challenging enough to test reasoning skills and being solvable by smaller models.
- **Distinct from Traditional Math**: It deviates from standard math problems, making it a valuable tool for assessing a model's ability to generalize and adapt.
- **Existing Projects**: TinyZero and Mini-R1 already provide a baseline with detailed code and training costs. It would make sense to stick with this problem for enhancing a distill to provide a good base of comparison.

## Challenges in Training a Reasoning Distill Model

Training the distilled model to perform reasoning tasks presented several challenges:

### Understanding System Prompts

The small 1.5b model struggled to comprehend the system prompt that required it to encapsulate its reasoning within `<think>` tags and its final answer within `<answer>` tags. This limitation necessitated a different approach.

### Model-Specific Output Preferences

DeepSeek distills have a preference for outputting formatted LaTeX when presented with mathematical problems. While the naive approach expected answers like `<answer>4/4+1</answer>`, the model would often produce outputs like `\boxed{\frac{4}{4}+1=2}`.

## Designing Effective Reward Functions

To guide the model's learning process, I crafted reward functions to work with the model's natural tendencies rather than against them. I suspect this will become a very important skill for the open-source community in the future. To do this the prompt was changed to one recommended by DeepSeek and the reward functions re-written to extract from a `\boxed{}` and evaluate LaTeX.

Prompt:

```
Using the numbers {numbers} and the basic arithmetic operations, write down an equation that equals {target}. Each number can only be used once.

Please reason step by step, and put your final answer within \boxed{}.
```

To accommodate the model's preference for outputting LaTeX equations, I utilized the excellent [math-verify](https://github.com/huggingface/Math-Verify) library, which is adept at handling and verifying LaTeX-formatted mathematical expressions.

```
def verify_ans(text, target, nums):
    # Extract the boxed final answer
    boxed = extract_last_boxed(text)
    if not boxed:
        return 0.0
    
    # Parse the boxed equation
    spl = boxed.split("=")
    return max([check_ans(x.strip(), target, nums) for x in spl])

def check_ans(text, target, nums):
    try:
        # Extract the numbers used in the proposed answer
        answer = parse(f"${text}$")
        gold = parse(str(target))
        nums_used = extract_numbers(text)
        
        # Check if all numbers are used exactly once
        if sorted(nums_used) != sorted(nums):
            return 0.0
        
        # Validate the equation's correctness
        if verify(gold, answer):
            return 1.0
    except:
        return 0.0
    return 0.0
```

## Training Process and Results

![reward1.png](attachment:reward1.png)

![completion1.png](attachment:completion1.png)

The following training parameters are suitable for training on a single H100 with FFT (full fine tuning).

```
training_args = GRPOConfig(
    use_vllm=True,  # Enable vLLM for fast inference
    vllm_device="cuda:0",
    vllm_gpu_memory_utilization=0.08,
    vllm_max_model_len=4096 + 64,  # Sequence + prompt length
    vllm_dtype="auto",
    learning_rate=5e-7,
    beta=0.001,
    temperature=0.6,
    lr_scheduler_type="cosine",
    per_device_train_batch_size=8,
    num_generations=8,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    max_steps=150,
    save_steps=100,
    output_dir="output_exp2",
)
```

### Initial Performance

Before applying GRPO, the model achieved a success rate of approximately 36% on the countdown task. This baseline performance indicated that the model possessed some inherent reasoning capabilities.

### Training Result

Using GRPO, I trained the model for 100 steps, which resulted in a significant improvement in the success rate:

- **52.4% Success Rate**: The model demonstrated a remarkable increase in its ability to solve countdown problems, reaching a success rate comparable to that of TinyZero and Phil Schmid's models, which were trained from scratch.

### Key Observations

- **Reward Increase and Plateau**: The reward function showed a steady increase up to around 50%, after which it plateaued. Further training led to instability and, ultimately, a collapse in performance.
- **Sequence Length Constraint**: The unintended constraint of 4096 tokens limited the model's ability to generate longer, more detailed reasoning traces, which are often necessary for solving complex problems.
- **Fundamental Thinking Patterns**: The model's underlying reasoning approach remained consistent with its original behavior, suggesting that GRPO enhanced its reasoning capabilities without altering its fundamental thought processes.

### Training Insights

- **Instability**: The training process exhibited instability, particularly when the model was trained beyond 100 steps. This instability could be attributed to the sequence length constraint and the inherent limitations of the distilled model.
- **Resource Efficiency**: Training on a single H100 GPU took a few hours, making it a cost-effective approach for enhancing smaller models.

## Potential of LORA Training

While I initially used **FFT** for compatibility and performance reasons, I believe that **LORA (Low-Rank Adaptation)** training holds promise for the following reasons:

- **Stability**: LORA affects a smaller subset of parameters, potentially leading to more stable training.
- **Domain-Specific Enhancement**: It may be particularly effective for domain-specific GRPO enhancement of existing models.
- **Performance**: Although my initial trials with LORA on a 4090 GPU showed promising results, the process was extremely slow. I found some issues with LORA 

## Conclusion

The application of GRPO to the DeepSeek R1 Distill-Qwen-1.5B model demonstrates the potential of reinforcement learning in enhancing the reasoning capabilities of distilled language models. While the results are promising, with the model achieving a 52.4% success rate on the countdown task, several challenges remain:

- **Sequence Length Constraint**: Addressing the limitation of 4096 tokens could lead to further improvements.
- **Model Stability**:  Further research is needed to understand and mitigate the instability observed during training.
- **LORA Optimization**: Exploring LORA training in more depth could unlock additional performance gains.

## Future Directions

Looking ahead, I am excited about the following possibilities:

- **Scaling Up**:  Investigating the impact of training on larger models and longer sequences.
- **Exploring Other Domains**:  Applying GRPO to other reasoning tasks beyond the countdown problem.
- **Optimizing LORA**:  Fine-tuning LORA training to enhance its effectiveness and efficiency.

## Acknowledgments

I would like to thank the open-source community for providing the tools and resources that made this research possible. Special thanks to the developers of DeepSeek, TinyZero, and the math-verify library.

## References

- [TinyZero GitHub Repository](https://github.com/Jiayi-Pan/TinyZero)
- [DeepSeek R1 Paper](https://arxiv.org/abs/2501.12948)
- [Group Relative Policy Optimization (GRPO) Paper](https://arxiv.org/abs/2402.03300)
- [math-verify Library](https://github.com/sympy/sympy)
