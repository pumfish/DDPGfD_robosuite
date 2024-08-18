import torch.nn as nn


class ActorNet(nn.Module):
    def __init__(self, in_dim, out_dim, device):
        super(ActorNet, self).__init__()
        self.device = device
        self.net = nn.Sequential(nn.Linear(in_dim, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(),
                                 nn.Linear(512, out_dim), nn.Tanh())  # +-1 output

    def forward(self, state):
        """
        :param state: N, in_dim
        :return: Action (deterministic), N,out_dim
        """
        action = self.net(state)
        # import time

        # print(f"state device is {state.device}")
        # print(f"self.device is {self.device}")

        # t1 = time.time()
        # print(f"shape = {state.shape}")
        # state = self.net[0](state)
        # t2 = time.time()
        # print(f"shape = {state.shape}")
        # state = self.net[1](state)
        # t3 = time.time()
        # print(f"shape = {state.shape}")

        # state = self.net[2](state)
        # t4 = time.time()
        # print(f"shape = {state.shape}")

        # state = self.net[3](state)
        # t5 = time.time()
        # print(f"shape = {state.shape}")

        # state = self.net[4](state)
        # t6 = time.time()
        # print(f"shape = {state.shape}")

        # action = self.net[5](state)
        # t7 = time.time()


        # print(f" 1 linear cost {t2-t1}s\n",
        #       f"1 relu cost {t3-t2}s\n",
        #       f"2 linear cost {t4-t3}s\n",
        #       f"2 relu cost {t5-t4}s\n",
        #       f"3 linear cost {t6-t5}s\n",
        #       f"1 tanh cost {t7-t6}s")
        # print("==="*10)
        return action


class CriticNet(nn.Module):
    def __init__(self, s_dim, a_dim, device):
        super(CriticNet, self).__init__()
        self.device = device
        in_dim = s_dim + a_dim
        self.net = nn.Sequential(nn.Linear(in_dim, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(),
                                 nn.Linear(512, 1))

    def forward(self, sa_pairs):
        """
        :param sa_pairs: state-action pairs, (N, in_dim)
        :return: Q-values , N,1
        """
        return self.net(sa_pairs)
