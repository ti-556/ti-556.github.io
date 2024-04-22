---
title: 'Vanilla Policy Gradient (VPG)'
date: 2024-04-23
permalink: /posts/2024/04/vanilla-policy-gradient/
tags:
  - Reinforcement Learning
  - Machine Learning
---
<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax:{inlineMath:[['\$','\$'],['\\(','\\)']],processEscapes:true},CommonHTML: {matchFontHeight:false}});</script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>


Introduction
======
Value-based methods like Q-learning have an intermediate step of learning a value function to optimize policy. With policy-based methods, we can skip this step and directly optimize the policy. 

If you know how neural networks learn, the idea behind policy gradient is very simple. As the name suggests, we’re going to use a similar approach to stochastic gradient descent to search for the optimal policy.

Characteristics
------
- on-policy - the learned policy is used to select the agent’s action
- can be used for both discrete and continuous action spaces

Loss Function
------
Like how we defined the loss function for SGD, we need a function that measures the performance of a policy. We call this the objective function, and this gives us the expected cumulative reward given a specific trajectory (a sequence of actions and states).

This can be calculated quite intuitively:
\\[
\boxed{J(\theta) = \Sigma_{\tau}P(\tau ; \theta)R(\tau)}
\\]
where $R(\tau) = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots$

The function is simply getting the product of the trajectory’s reward $R(\tau)$ and its probability $P(\tau ; \theta)$ under policy $\theta$, then taking the sum over all trajectories.

Policy Gradient
------
Now, we want to get the gradient of the objective function. The policy gradient can be calculated with the following equation:
\\[
\nabla_{\theta}J(\theta) = \mathop{\mathbb{E}}_{\pi_{\theta}}[\Sigma_{t=0}^{T}\nabla_{\theta}log\pi_{\theta}(a_t|s_t)R(\tau)]
\\]
where $\pi_{\theta}(a_t|s_t)$ is the probability of taking action $a_t$ from state $s_t$.

The detailed derivation of this policy gradient function can be found [here](https://huggingface.co/learn/deep-rl-course/unit4/pg-theorem)

Reward-to-go
------
The policy gradient function looks right, but let’s think about it. Essentailly, we’re trying to update the policy parameters by this gradient (SGD), so according to this function, we’re going to change the policy proportionally to the trajectory’s cumulative reward $R(\tau)$. This cumulative reward is going to be used to adjust the probability of an action being taken at a certain state. However, does this really accurately measure whether the action was good or not?

We don’t really care about anything before when measuring how well an action is. What matters is everything after the action, in other words, the action’s consequence.

Considering this, an alternative measurement is used instead of the cumulative reward. This is called the reward-to-go:
\\[
\hat{R}_t = \Sigma_{t'=t}^{T}R(s_{t'}, a_{t'}, s_{t'+1})
\\]
So, we can rewrite the policy gradient function as:
\\[
\nabla_{\theta}J(\theta) = \mathop{\mathbb{E}}_{\pi_{\theta}}[\nabla_{\theta}log\pi_{\theta}(a_t|s_t)\hat{R}_t]
\\]

Baseline
------
Any function that doesn’t depend on action is called a baseline $b(s_t)$. We subtract the reward-to-go with the baseline in order to reduce variance. Now, the policy gradient function can be written as:
\\[
\nabla_{\theta}J(\theta) = \mathop{\mathbb{E}}_{\pi_{\theta}}[\nabla_{\theta}log\pi_{\theta}(a_t|s_t)\Sigma_{t'=t}^{T}(R(s_{t'}, a_{t'}, s_{t'+1}) - b(s_t))]
\\]
First of all, why is this possible? Wouldn’t subtracting from the policy gradient function make it invalid?

We’re able to subtract the policy gradient with the baseline due to the fact that the “expected value of the gradient of the log of a probability distribution” is equivalent to zero. Ok, that’s long, let’s abbreviate this as the Expected Grad-Log-Prob (EGLP) Lemma:
\\[
\mathop{\mathbb{E}}[\nabla_{\theta} logP_{\theta}(x)] = 0
\\]