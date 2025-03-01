# Enhancing Reasoning in Distilled Language Models with GRPO

## Introduction

Since the release of DeepSeek-R1, there has been huge community interest in applying GRPO (Group Relative Policy Optimization) to reasoning models. However, large models like DeepSeek-R1 are very computationally expensive to train from scratch and so the open-source community has exploded with large numbers of proof of concepts and examples based on small models. [TinyZero](https://github.com/Jiayi-Pan/TinyZero) and [Mini-R1](https://www.philschmid.de/mini-deepseek-r1) are examples showing that small 3b models can learn complex reasoning directly from scratch using GRPO.

It has also been shown that reasoning capabilities can be learned for smaller models by using filtered traces of larger models [S1](https://huggingface.co/simplescaling/s1.1-32B). This appears to have several advantages, firstly it is way cheaper than training a model like DeepSeek-R1 from scratch and secondly the smaller model is then cheaper to inference and can be used on far cheaper hardware than DeepSeek-R1 which requires an optimized datacentre deployment.

Unfortunately, distilled models suffer from some drawbacks. They do not seem to retain all the capabilities of their teacher models and have gaps in their capabilities. It has been proposed originally by [DeepSeek](https://arxiv.org/abs/2501.12948) that taking a reasoning model as a base and then applying further GRPO to it could yield good results.

In this article, I will share my experience of using GRPO to enhance the reasoning capabilities of **DeepSeek-R1-Distill-Qwen-1.5B** on a toy problem to demonstrate it is not just possible to enhance existing performance [DeepScaleR](https://github.com/agentica-project/deepscaler) but that it is possible to teach it new reasoning domains as well, to some extent.

## Toy Reasoning

The countdown problem, popularized by [TinyZero](https://github.com/Jiayi-Pan/TinyZero), is an excellent toy problem to study. Here's why:

- **Not Too Difficult**: It strikes a balance between being challenging enough to test reasoning skills and being solvable by smaller models.
- **Distinct from Traditional Math**: It deviates from standard math problems, making it a valuable tool for assessing a model's ability to generalize and adapt.
- **Existing Projects**: TinyZero and Mini-R1 already provide a baseline with detailed code and training costs. It would make sense to stick with this problem for enhancing a distill to provide a good base of comparison.

In this version of the countdown problem, we stick to the version introduced by Jiayi-Pan where the assistant has to reach a target number (e.g. 2) by combining a list of other numbers (e.g. 4, 4, 1) using basic arithmetic operations (+, -, *, /).

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

![reward1.png](/reward1.png)

### Initial Performance

Before applying GRPO, the model achieved a success rate of approximately 36% on the countdown task. This baseline performance indicated that the model possessed some inherent reasoning capabilities.

### Training Result

Using GRPO, I trained the model for 100 steps, which resulted in a significant improvement in the success rate:

- **52.4% Success Rate**: The model demonstrated a remarkable increase in its ability to solve countdown problems, reaching a success rate comparable to that of TinyZero and Phil Schmid's models, which were trained from scratch.

### Key Observations

- **Reward Increase and Plateau**: The reward function showed a steady increase up to around 50%, after which it plateaued. Further training led to instability and, ultimately, a collapse in performance.
- **Sequence Length Constraint**: The unintended constraint of 4096 tokens limited the model's ability to generate longer, more detailed reasoning traces.
- **Fundamental Thinking Patterns**: The model's underlying reasoning approach remained consistent with its original behavior, suggesting that GRPO enhanced its reasoning capabilities without altering its fundamental thought processes.

To illustrate the difference here consider these example traces:

<details>
    <summary>Phil Schmid's Mini-R1</summary>

```
<think> We need to use the numbers 75, 99, 62, and 34 with basic arithmetic operations to get 72. Let's try different combinations:
- 99 - 75 + 62 - 34 = 92 - 34 = 58 (too high)
- 99 - 75 - 62 + 34 = 24 - 62 + 34 = -38 + 34 = -4 (too low)
- 99 + 75 - 62 - 34 = 174 - 96 = 78 (too high)
- 99 + 75 - 62 + 34 = 174 + 34 - 62 = 208 - 62 = 146 (too high)
- 99 - 75 + 62 + 34 = 24 + 96 = 120 (too high)
- 75 + 99 - 62 - 34 = 174 - 96 = 78 (too high)
- 75 + 99 - 62 + 34 = 174 + 34 - 62 = 208 - 62 = 146 (too high)
- 75 + 62 - 99 + 34 = 137 - 99 + 34 = 38 + 34 = 72
So, 75 + 62 - 99 + 34 equals 72.
</think>
<answer> 75 + 62 - 99 + 34 </answer>
```

</details>

<details>

<summary>This model</summary>

```
Alright, so I have this problem where I need to use the numbers 51, 80, and 96 along with basic arithmetic operations to make an equation that equals 67. Each number can only be used once. Hmm, okay, let me think about how to approach this.

First, let me write down the numbers: 51, 80, 96. I need to combine them using addition, subtraction, multiplication, division, and maybe exponents or parentheses if needed, but the problem says basic operations, so probably just the four I mentioned.

My goal is to get from 51, 80, 96 to 67. So, maybe I can subtract or add these numbers in some way. Let me see.

Let me start by considering how close 67 is to the numbers given. 67 is less than 80, so maybe subtracting something might work. Let me try subtracting 51 from 80. 80 minus 51 is 29. Hmm, okay, so if I can somehow get 29 from the remaining number, which is 96, maybe I can divide 96 by something? Wait, 96 is a big number, so maybe dividing maybe by 2? But I don't see where 2 would come from unless I subtract something else.

Wait, let me think differently. Maybe instead of subtracting 51 from 80, I can combine all three numbers somehow. Hmm. Let me try to think about the operations.

What if I subtract 51 from 80? That gives me 29, as I thought earlier. Then, if I can somehow get 29 from 96. Hmm, 96 divided by something would give me 29. Let me see: 96 divided by 3 is 32, which is too big. 96 divided by 2 is 48, still too big. Maybe 96 minus something equals 29. Let me calculate: 96 minus what equals 29? That would be 96 - 29 = 67. Wait, that's not helpful because that's the same number. Hmm.

Alternatively, maybe if I can subtract 51 from 96. 96 minus 51 is 45. So now I have 45 and how can I use 80? Hmm, 80 minus 45 is 35? Hmm, that's not helpful either.

Alternatively, let's think about the numbers more. Maybe if I can use multiplication or division. For example, let me see if any two numbers can divide into the third.

Looking at 51 and 80: 51 * 80 is 4080, which is way too big. 51 divided by 80 is about 0.6375. 80 divided by 51 is about 1.568. Not helpful.

How about multiplication? If I multiply two numbers and then subtract or add the third? Maybe 51 * 80 is 4080, then subtracting 96 would be 4080 - 96 = 3984, which is way too big. Dividing 4080 by 96 is 42.5, which is still not 67.

Hmm. Maybe division is a better approach. Let me try dividing two numbers and then subtracting or adding the third.

What if I take 96 divided by something? Let me think. 96 divided by 1.5 is 64, but I don't have 1.5. Alternatively, maybe 96 divided by something to get a number that when subtracted from another number gives me 67.

Wait, if I subtract a larger number, but 96 is the largest number, so I can't get bigger than that. Wait, unless I'm subtracting from another number. Hmm.

Wait, perhaps I can use subtraction in a way that the numbers are added or subtracted to get 67.

Let me consider the numbers in terms of equations.

I need to make 67. So, perhaps:

51 + something = 67

But 67 - 51 is 16, so I need to get 16 from 80 and 96.

Similarly, 80 - something = 67, so something would be 13, but I don't think I can get 13 from 96.

Alternatively, 96 - something = 67, so something is 29. Then, how can I get 29 from 80 and 51? Maybe subtract 80 and 51.

Wait, wait, maybe I can think of it as 80 - 51 = 29, which is what we had earlier. So then, if I have 96 - (80 - 51) that would be 96 - 29 = 67.

Let me double-check this:

96 - (80 - 51) = ?

First, compute the innermost parentheses: 80 - 51 = 29

Then, subtract that from 96: 96 - 29 = 67

Yes! That works.

So, the equation is 96 - (80 - 51) = 67

Let me verify:

Compute the parentheses first: 80 - 51 = 29

Then, subtract that from 96: 96 - 29 = 67

Yes, that's correct.

So, the equation using each number once is 96 minus (80 minus 51) equals 67.

Let me double-check that all numbers are used once:

- 96 is used once
- 80 is used once
- 51 is used once

And all operations are basic arithmetic: subtraction and parentheses.

Therefore, the equation is correct.

**Final Answer**
The equation is \boxed{96 - (80 - 51) = 67}.
</think>

To solve the problem of using the numbers 51, 80, and 96 with basic arithmetic operations to equal 67, we can follow these steps:

1. First, subtract 51 from 80:
   \[
   80 - 51 = 29
   \]

2. Then, subtract the result from 96:
   \[
   96 - 29 = 67
   \]

Thus, the equation that equals 67 using the numbers 51, 80, and 96 is:
\[
96 - (80 - 51) = 67
\]

The final answer is:
\[
\boxed{96 - (80 - 51) = 67}
\]
67
```

</details>

While the first model has learned an effective strategy for solving these particular problems, it is unlikely that this would generalize to other tasks. The second model, on the other hand, has learned to enhance its already inherent reasoning capabilities to better solve the target class of problems. 

### Training Insights

![completion1.png](/completion1.png)

The following training parameters are suitable for training on a single H100 with FFT (full fine tuning). The training takes a few hours making experiments like this very cost-effective.

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

### Instability

The training process exhibited instability after 100 steps of training. This instability persisted with various choices of parameters and I think it could be attributed to forcing the model to answer within 4096 tokens contrary to how it was originally trained. This is an unintended constraint which is coming from resource issues. It is also possible that very small models have a different optimal way of working through problems to that which is taught by large models (such as the Mini-R1 example). Embracing that however, if true, would require much longer training and destroy the model's general reasoning capabilities.

### Potential for LORA Training

While I initially used **FFT** for compatibility and performance reasons, I believe that **LORA (Low-Rank Adaptation)** training holds promise for the following reasons:

- **Stability**: LORA affects a smaller subset of parameters, potentially leading to more stable training.
- **Domain-Specific Enhancement**: It may be particularly effective for domain-specific GRPO enhancement of existing models.
- **Performance**: Although my initial trials with LORA on a 4090 GPU showed promising results, the process was extremely slow. Once LORA support has been optimized it should be much faster and more memory efficient than FFT, though this was not the case when I ran these experiments.

## Conclusion & Future Work

The application of GRPO to the DeepSeek R1 Distill-Qwen-1.5B model demonstrates the potential of reinforcement learning in enhancing the reasoning capabilities of distilled language models. However, there are clear directions for improvement and future work:

- **Longer Sequence Length**: DeepSeek-R1 was trained with up to 32k for its reasoning traces, 4096 is a small fraction of this and is truncating model responses resulting in an artificial length constraint, possible instability and possible reduced performance. Longer sequences require far more resources needing more than one datacentre-class GPU for training.
- **LORA Optimization**: Once performant implementations of PEFT are available that work well with GRPO and DeepSpeed it is likely to be worth investigating the same class of experiments but with LORA instead. This could further reduce hardware and compute requirements.

This said though, the most promising future work after this is probably an entire pipeline involving a SFT stage on good high quality traces and a GRPO stage on community sourced verifier functions. This could be done on a much larger model, say 32b size. I believe that once the community has made good verifier functions and ironed out the details of training distill models we will be able to build a fully open source medium size high quality reasoning model.

If you made it to the end, I think this is the highest impact thing we can do. It is also surprisingly difficult and interesting as my own example functions above illustrate.

## Acknowledgments

I would like to thank the open-source community (especially @Azure and @MrDragonFox on Discord) for providing the information, helpful discussions, tools and resources that made this research possible. Special thanks to the developers of DeepSeek!

## References

- [TinyZero GitHub Repository](https://github.com/Jiayi-Pan/TinyZero)
- [DeepSeek R1 Paper](https://arxiv.org/abs/2501.12948)
- [Group Relative Policy Optimization (GRPO) Paper](https://arxiv.org/abs/2402.03300)
- [math-verify Library](https://github.com/huggingface/Math-Verify)
