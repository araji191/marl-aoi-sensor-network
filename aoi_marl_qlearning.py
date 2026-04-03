import random

random.seed(0)

# ===PARAMETERS===

NUM_SENSORS      = 8

# Actions
WAIT             = 0
TRANSMIT         = 1
ACTIONS          = [WAIT, TRANSMIT]

# Transmission status
IDLE             = 0
SUCCESS          = 1
COLLISION        = 2

# Q-learning
gamma            = 0.75
alpha            = 0.05
num_time_slots   = 100
num_episodes     = 1000 * NUM_SENSORS

# Exploration
epsilon_start    = 1.0
epsilon_end      = 0.05
epsilon_decay    = (epsilon_end / epsilon_start) ** (1 / num_episodes)

# AoI
A_MAX            = NUM_SENSORS * 4

# Evaluation
num_eval_episodes = 1000

class Sensor:
    def __init__(self, power_budget):
        self.power_budget = power_budget
        self.battery = Battery(power_level=power_budget)
        self.time_since_last_tx = 0
        self.last_outcome = IDLE
        self.q_table = {}

    def get_reward(self, aoi):
            if self.last_outcome == SUCCESS:
                base_reward = 2.0
            elif self.last_outcome == COLLISION:
                base_reward = -2.0
            else: # IDLE
                base_reward = -0.5

            aoi_penalty = -2.0 * (min(aoi, A_MAX) / A_MAX) 

            total_reward = base_reward + aoi_penalty
            
            return total_reward
        
    def reset(self):
        self.time_since_last_tx = 0

    def age(self):
        self.time_since_last_tx += 1

    def get_state(self):
        return (min(self.time_since_last_tx, A_MAX), self.last_outcome, self.battery.power_level)
    
    def choose_action(self, state, epsilon):
        available = [WAIT] if self.battery.power_level <= self.battery.T2 else ACTIONS

        if random.random() < epsilon:
            return random.choice(available)

        q_vals = {a: self.get_q(state, a) for a in available}
        max_q = max(q_vals.values())

        best = [a for a, q in q_vals.items() if q == max_q]
        return random.choice(best)

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def set_q(self, state, action, value):
        self.q_table[(state, action)] = value

class Battery:
    def __init__(self, power_level):
        self.max_level = power_level
        self.power_level = power_level
        self.T1 = power_level // 2
        self.T2 = 1

    def recharge(self):
        if self.power_level <= self.T1:
            self.power_level = min(self.power_level + 1, self.max_level)

class Monitor:
    def __init__(self, num_sensors):
        self.num_sensors = num_sensors
        self.time = 0
        self.last_received_time = [0] * num_sensors

    def set_time(self, time):
        self.time = time

    def update_on_arrival(self, sensor_id, generation_time):
        self.last_received_time[sensor_id] = generation_time

    def get_aoi(self, sensor_id):
        return max(0, self.time - self.last_received_time[sensor_id])

    def get_all_aoi(self):
        return [self.get_aoi(i) for i in range(self.num_sensors)]

    def get_avg_aoi(self):
        return sum(self.get_all_aoi()) / self.num_sensors

class WirelessSensorNetwork:
    def __init__(self, num_sensors):
        self.num_sensors = num_sensors
        self.time = 0
        self.monitor = Monitor(num_sensors=num_sensors)
        self.sensors = (
            [Sensor(power_budget=20) for _ in range(num_sensors // 2)] +
            [Sensor(power_budget=10) for _ in range(num_sensors - (num_sensors // 2))]
        )
        self.past_rewards = []
        self.in_flight_packets = []  

    def step(self, actions):
        self.time += 1
        self.monitor.set_time(self.time)

        for packet in self.in_flight_packets[:]:
            arrival_time, sensor_id, generation_time = packet
            if arrival_time <= self.time:
                self.monitor.update_on_arrival(sensor_id, generation_time)
                self.in_flight_packets.remove(packet)

        for i, sensor in enumerate(self.sensors):
            if actions[i] == TRANSMIT and sensor.battery.power_level <= sensor.battery.T2:
                actions[i] = WAIT

        states = [s.get_state() for s in self.sensors]
        transmitting = [i for i, a in enumerate(actions) if a == TRANSMIT]

        if len(transmitting) == 1:
            idx = transmitting[0]
            delay = random.randint(1, 3)
            arrival_time = self.time + delay
            generation_time = self.time

            self.in_flight_packets.append((arrival_time, idx, generation_time))

            self.sensors[idx].last_outcome = SUCCESS
            self.sensors[idx].battery.power_level -= 1
            self.sensors[idx].time_since_last_tx = 0 

            for j, sensor in enumerate(self.sensors):
                if j != idx:
                    sensor.last_outcome = IDLE

        elif len(transmitting) == 0:
            for sensor in self.sensors:
                sensor.last_outcome = IDLE

        else:
            tx_set = set(transmitting)
            for j, sensor in enumerate(self.sensors):
                if j in tx_set:
                    sensor.last_outcome = COLLISION
                    sensor.battery.power_level = max(0, sensor.battery.power_level - 1)
                else:
                    sensor.last_outcome = IDLE

        for j, sensor in enumerate(self.sensors):
            if not (len(transmitting) == 1 and j == transmitting[0]):
                sensor.age()

    
        for sensor in self.sensors:
            sensor.battery.recharge()

        next_states = [s.get_state() for s in self.sensors]

        rewards = []
        for i, sensor in enumerate(self.sensors):
            aoi = self.monitor.get_aoi(i)
            r = sensor.get_reward(aoi)
            rewards.append(r)

        step_reward = sum(rewards)

        for i, sensor in enumerate(self.sensors):
            s = states[i]
            a = actions[i]
            r = rewards[i]
            sp = next_states[i]

            current_q = sensor.get_q(s, a)
            
            available_next = [WAIT] if sensor.battery.power_level <= sensor.battery.T2 else ACTIONS
            max_next_q = max(sensor.get_q(sp, ap) for ap in available_next)

            new_q = current_q + alpha * (r + gamma * max_next_q - current_q)
            sensor.set_q(s, a, new_q)

        return step_reward
    
    def reset(self):
        self.time = 0
        self.in_flight_packets = []  
        self.monitor = Monitor(num_sensors=self.num_sensors) 
        for sensor in self.sensors:
            sensor.reset()
            sensor.last_outcome = IDLE
            sensor.battery.power_level = sensor.power_budget

    def get_avg_aoi(self):
        return self.monitor.get_avg_aoi()

    def get_cumulative_reward(self):
        cumulative_reward = 0
        for i in range(self.num_sensors):
            aoi = self.monitor.get_aoi(i)
            cumulative_reward += self.sensors[i].get_reward(aoi)
        return cumulative_reward

def random_policy(sensors):
    actions = []
    for sensor in sensors:
        if sensor.battery.power_level <= sensor.battery.T2:
            actions.append(WAIT)
        else:
            actions.append(TRANSMIT if random.random() < 1 / len(sensors) else WAIT)
    return actions