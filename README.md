![Logo](./fruit/docs/images/home-logo.png)

**Fruit API** is a universal deep reinforcement learning framework, which is designed 
meticulously to provide a friendly user interface. Specifically, **Fruit API** has the 
following noticeable contributions:


* **Friendly API**: **Fruit API** follows a modular design combined with the OOP in Python
to provide a solid foundation and an easy-to-use user interface via a simplified 
API. Based on the design, our ultimate goal is to provide researchers a means to 
develop reinforcement learning (RL) algorithms with little effort. In particular, 
it is possible to develop a new RL algorithm under 100 lines of code. What users 
need is to create a `Config`, a `Learner`, and plug them into the framework. We
also provides a lot of sample `Config`s and `Learner`s in a hierarchical structure
so that users can inherit a suitable one.

![Figure 1](./fruit/docs/images/figure_1.png)

* **Portability**: The framework can work properly in different operating systems such as 
Windows, Linux, and Mac OS.

* **Interoperability**: We keep in mind that the framework should work with any deep learning
libraries such as PyTorch, Tensorflow, Keras, etc. Researchers would define the neural 
network architecture in the config file by using their favourite libraries. Instead of 
implementing a lot of deep RL algorithms, we provide a flexible way to integrate with 
existing deep RL libraries.

* **Generality**: The framework supports different disciplines in reinforement learning 
such as: multiple objectives, multiple agents, and human-machine interaction.

We implemented a set of deep RL baselines in different disciplines as sample references:

*Value-based deep RL*:

 * Deep Q-Network (DQN)
 * Double DQN
 * Duel DQN
 * Prioritized Experience Replay (Proportional-based approach)
 * DQN variants (asynchronous/synchronous method)
 
*Policy-based deep RL*:

 * A3C
 
*Multi-agent deep RL*:

 * Multi-agent A3C
 * Multi-agent A3C with communication map
 
*Multi-objective RL/deep RL*:

 * Q-Learning
 * Multi-objective Q-Learning (linear and non-linear method)
 * Multi-objective DQN (linear and non-linear method)
 * Multi-objective A3C (linear and non-linear method)
 * Single policy/multi-policy method
 * Hypervolume calculation
 
*Human-agent interaction*

 * A3C with map
 * Divide and conquer strategy with DQN
 
Finally, we developed 5 environments as a testbed to examine different disciplines in deep RL:
* Mountain car (multi-objective environment/graphical support)
* Deep sea treasure (multi-objective environment/graphical support)
* Tank battle (multi-agent/multi-objective/human-agent cooperation environment)
* Food collector (multi-objective environment)
* Milk factory (multi-agent/heterogeneous environment)
 
Demonstrations can be found here:

<div align="center">
  <a href="https://www.youtube.com/watch?v=WCa6n1F6UM8" target="_blank">
    <img src="http://img.youtube.com/vi/WCa6n1F6UM8/3.jpg"
         alt="Fruit API - Tank Battle"
         width="240" height="180" border="10" />
  </a>
  <a href="https://www.youtube.com/watch?v=eoud2D0nW1k" target="_blank">
    <img src="http://img.youtube.com/vi/eoud2D0nW1k/0.jpg"
         alt="Fruit API - Milk Factory"
         width="240" height="180" border="10" />
  </a>
  <a href="https://www.youtube.com/watch?v=usJP9Gr9nkM" target="_blank">
    <img src="http://img.youtube.com/vi/usJP9Gr9nkM/0.jpg"
         alt="DeepMind Lab - Laser Tag Space Bounce Level (Hard)"
         width="240" height="180" border="10" />
  </a>
  <br /><br />
</div>
 
 