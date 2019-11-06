https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html

# Policy Gradient

- Policy Gradient는 policy 그 자체를 직접 모델링하고 최적화한다.

- policy $\pi$는 파라미터 $\theta$를 갖는다.  $\pi_{\theta}(a|s)$  

- policy를 통해 다음 action이 결정되고 이에 따라 reward가 주어지므로, reward value function은 policy에 의존한다. 

- 즉, policy의 파라미터 $\theta$ 를 최적화하여 reward value를 극대화할 수 있다.

- reward function은 다음과 같다.

  - $J(\theta) =\sum_{s \in S} \textcolor{red}{d^{\pi}(s)}\textcolor{blue}{V^{\pi}(s)} = \sum_{s \in S} \textcolor{red}{d^{\pi}(s)} \sum_{a \in A} \pi_{\theta}(a|s)\textcolor{green}{Q^{\pi}(s, a)}$

  - $\textcolor{red}{d^{\pi}(s)}$: stationary distribution of Markov Chain for $\pi$. Markov chain 상의 state를 영원히 돌아다니다 보면 결국, 어떤 state를 방문할 확률은 어떤 값으로 수렴해 변하지 않게 된다. 즉 여기서는 어떤 state에 갈 확률.
    - $d^{\pi}(s) = \lim_{t \rightarrow \infin}P(s_t = s|s_0, \pi_\theta)$ 
    - $t$가 무한으로 갈때 $s_0$ 에서 시작해서 policy $\pi_\theta$를 이용해 $s_t$에 도달할 확률을 의미
    - Markov chain의 stationary distribution은 왜 PageRank Algorithm이 동작하는지에 대한 답임 (왜그럴까? Rabbit hole!) https://jeremykun.com/2015/04/06/markov-chain-monte-carlo-without-all-the-bullshit/
  - $\textcolor{blue}{V^{\pi}(s)}$: state $s$ 의 value 
    - $V^{\pi}(s) = \mathbb{E}_{a \sim \pi}[G_t|S_t=s]$
    - state $s$ 에서 취할 수 있는 action들의 가치의 기대값
  - $\textcolor{green}{Q^{\pi}(s, a)}$: state $s$ 와 action $a$ 페어의 value
    - $Q^{\pi}(s, a) = \mathbb{E}_{a \sim \pi}[G_t|S_t=s, A_t=a]$
  - 그래서 결국 $J(\theta)$는 각 state의 방문 확률에 그 value를 곱해 모두 더한 값이 된다. 

- policy-based method는 continuous한 action space를 가진 문제를 풀기에 좋다.

  - value-based method는 각 action의 value를 계산해야 하므로, continuous한 space를 discretize해야 하고, 계산량이 필연적으로 많아지는 문제가 존재.
  - gradient ascent를 이용하면, $\triangledown_{\theta}J(\theta)$가 이끄는 방향으로 $\theta$ 를 이동시켜 가장 큰 $J(\theta)$ 를 얻을 수 있는 $\theta$ 에 도달할 수 있다.



# Policy Gradient Theorem

-  $\triangledown_\theta J(\theta)$를 구하는 것이 어렵다. 왜? stationary distribution $\textcolor{red}{d^{\pi}(s)}$ 와 action selection $\textcolor{green}{Q^{\pi}(s, a)}$ 두가지에 모두 디펜던시가 걸려있다. 
  - env에 대한 정보가 없으면 policy 업데이트로 인한 state distribution의 효과를 추정하기 어렵다.
  - 그래서 여기서 policy gradient theorem을 사용한다!
- $\triangledown_\theta J(\theta)$ 를 아래와 같이 변환하면, $d^{\pi}(s)$의 미분값을 이용하지 않아도 되고, 계산도 간단해진다.
  - $\triangledown_{\theta}J(\theta) = \textcolor{red}{\triangledown_\theta} \sum_{s \in S} d^{\pi}(s) \sum_{a \in A} Q^{\pi}(s, a)\pi_{\theta}(a|s)$
  - $ \propto \sum_{s \in S} d^{\pi}(s) \sum_{a \in A}Q^{\pi}(s, a)\textcolor{red}{\triangledown_{\theta}}\pi_{\theta}(a|s)$ 
  - gradient가 전체가 아닌 action selection 부분에만 붙도록 바뀌었다. 왜 그럴까?



# Proof of Policy Gradient Theorem

와.. 이건 길다. Rabbit Hole!



# Policy Gradient Algorithm

## REINFORCE (Monte-Carlo policy gradient)

- $\theta$를 업데이트 하기 위해 Monte-Carlo 방법을 이용해서 episode를 샘플링하고 return을 추정한다.

- Sample gradient의 기대값이 실제 gradient와 같기 때문에 REINFORCE가 작동한다.

  - $\triangledown_\theta J(\theta) = \mathbb{E}_\pi [Q^\pi(s, a) \triangledown_\theta \ln \pi_\theta(a|s)]$
  - $= \mathbb{E}_\pi [G_t \triangledown_\theta \ln \pi_\theta (A_t | S_t)]$
  - 왜냐면 $Q^\pi(S_t, A_t) = \mathbb{E}_\pi[G_t|S_t, A_t]$

- 그래서 real sample trajectories로부터 $G_t$를 측정할 수 있고, 이것을 이용해서 policy gradient를 업데이트할 수 있다.

- 이때 전체 full trajectory를 생성하기 때문에 이를 Monte-Carlo 방법이라 한다.

- 프로세스는 다음과 같다.

  - 1. policy parameter $\theta$ 를 초기화한다.

    2. $\pi_\theta$ 로부터 하나의 trajectory를 생성한다.

    3. $t=0, 1 ... , T$까지 loop를 돌면서

       1. return $G_t$를 추정한다.

    4. 파라미터 $\theta$ 를 업데이트한다.

       $\hat{g} = \Sigma_{t=0}^T \triangledown_{\theta}\ln\pi_{\theta}(a_t | s_t)G_t$

       $\theta \leftarrow \theta + \alpha \hat{g} $ 

- 팁

  - 널리 사용되는 REINFORCE 버전은 $G_t$에서 baseline 값을 뺌으로써, bias는 유지한채로 gradient estimation의 variance는 줄이는 방식을 쓴다. 
    - 가장 많이 쓰이는 baseline은 action-value로부터 state-value를 빼는 것. 즉, advantage $A(s, a) = Q(s, a) - V(s)$ 로 gradient ascent를 한다. 
    - 왜 variance가 줄어드는걸까? https://danieltakeshi.github.io/2017/03/28/going-deeper-into-reinforcement-learning-fundamentals-of-policy-gradients/