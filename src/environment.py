import numpy as np
from gym import Env
from gym.spaces import Discrete, Box

import pygame

from pathlib import Path
abs_path = Path(__file__).parent

class TreasureRoom(Env):

    def __init__(self, N, verbose=0, change_room=False, room_disposition=None, deep=False):
        self.N = N
        self.size = self.N ** 2

        # verbosity
        self.verbose = verbose

        # room elements encoding
        self.agent_label = 0
        self.wardrobe_label = 1
        self.poison_label = 2
        self.treasure_label = 3

        # Actions we can take:
        # 0. UP
        # 1. LEFT
        # 2. RIGHT
        # 3. DOWN
        self.action_space = Discrete(4)

        # initialize room
        if room_disposition is None:
            self.agent_pos, self.wardrobe_pos, self.poison_pos, self.treasure_pos = self.__init__room__()
        else:
            self.wardrobe_pos, self.poison_pos, self.treasure_pos = room_disposition
            self.agent_pos = self.__init_agent_position__()

        # Observation space
        self.deep = deep
        self.change_room = change_room
        if not self.deep:
            self.observation_space = Box(low=0, high=self.size - 1, shape=(4,), dtype=np.int32)
        else:
            if not self.change_room:
                self.observation_space = Box(low=0, high=1, shape=(self.size,), dtype=np.int32)
            else:
                self.observation_space = Box(low=0, high=1, shape=(self.size, 5,), dtype=np.int32)

        # render object
        self.episode_count = 0
        self.timestep = 0
        self.room = self.__get_room__()

        # pygame gui utils
        self.fps = 4
        self.window_size = (min(64 * N, 512), min(64 * N, 512))
        self.cell_size = (
            self.window_size[0] // self.N,
            self.window_size[1] // self.N,
        )
        self.window_surface = None
        self.clock = None

        self.poison_img = None
        self.wardrobe_img = None
        self.agent_img = None
        self.treasure_img = None
        self.empty_cell_img = None
        self.win_img = None
        self.lose_img = None


    def step(self, action):

        self.timestep += 1

        reward = 0
        done = False

        # take action
        if self.is_adj(action) and not self.is_wardrobe(action):
            if action == 0:
                self.agent_pos -= self.N
            elif action == 1:
                self.agent_pos -= 1
            elif action == 2:
                self.agent_pos += 1
            elif action == 3:
                self.agent_pos += self.N

        # calculate reward and check if it is done
        if self.agent_pos == self.treasure_pos:
            reward += 10
            done = True
        elif self.agent_pos == self.poison_pos:
            reward -= 10
            done = True
        else:
            reward -= 1

        if self.deep:
            observation = self.__get_encoded_room__()

        else:
            observation = [self.agent_pos, self.wardrobe_pos, self.poison_pos, self.treasure_pos]

        return observation, reward, done, {}

    def reset(self, **kwargs):
        self.timestep = 0

        if self.deep:
            self.episode_count += 1

        if not self.change_room:
            self.agent_pos = self.__init_agent_position__()
        else:
            self.agent_pos, self.wardrobe_pos, self.poison_pos, self.treasure_pos = self.__init__room__()

        if self.deep:
            state = self.__get_encoded_room__()
        else:
            state = [self.agent_pos, self.wardrobe_pos, self.poison_pos, self.treasure_pos]

        return state

    def render(self, mode='text', **kwargs):

        if mode == 'text':
            # print timestep
            print(f'\nEpisode {self.episode_count}')
            print(f'Timestep {self.timestep}')

            # get current room disposition
            self.room = self.__get_room__()

            # display upper separators
            self.print_separators()
            print()

            # display room content
            for row in range(self.room.shape[0]):
                print('|', end='')
                for column in range(self.room.shape[1]):
                    element = self.room[row][column]
                    self.print_element(element)
                print()

            # display lower separators
            self.print_separators()
            print()
        elif mode == 'gui':
            self.render_gui()

    def render_gui(self):
        if self.window_surface is None:
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption("Treasure Room")
            self.window_surface = pygame.display.set_mode(self.window_size)

        assert (
            self.window_surface is not None
        ), "Something went wrong with pygame. This should never happen."

        if self.clock is None:
            self.clock = pygame.time.Clock()

        if self.poison_img is None:
            file_name = abs_path / "../img/skull_and_crossbones.png"
            self.poison_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.wardrobe_img is None:
            file_name = abs_path / "../img/construction.png"
            self.wardrobe_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.agent_img is None:
            file_name = abs_path / "../img/man-walking.png"
            self.agent_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.treasure_img is None:
            file_name = abs_path / "../img/moneybag.png"
            self.treasure_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.empty_cell_img is None:
            file_name = abs_path / "../img/white_large_square.png"
            self.empty_cell_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.win_img is None:
            file_name = abs_path / "../img/money_mouth_face.png"
            self.win_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.lose_img is None:
            file_name = abs_path / "../img/coffin.png"
            self.lose_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )

        caption = "Treasure Room - Episode {}".format(self.episode_count)
        pygame.display.set_caption(caption)

        for y in range(self.N):
            for x in range(self.N):
                pos = (x * self.cell_size[0], y * self.cell_size[1])
                rect = (*pos, *self.cell_size)

                cell_number = x * self.N + y 

                self.window_surface.blit(self.empty_cell_img, pos)
                if cell_number == self.wardrobe_pos:
                    self.window_surface.blit(self.wardrobe_img, pos)
                elif cell_number == self.poison_pos:
                    if cell_number == self.agent_pos:
                        self.window_surface.blit(self.lose_img, pos)
                    else:
                        self.window_surface.blit(self.poison_img, pos)
                elif cell_number == self.treasure_pos:
                    if cell_number == self.agent_pos:
                        self.window_surface.blit(self.win_img, pos)
                    else:
                        self.window_surface.blit(self.treasure_img, pos)
                elif cell_number == self.agent_pos:
                    self.window_surface.blit(self.agent_img, pos)

                pygame.draw.rect(self.window_surface, (180, 200, 230), rect, 1)
                # filename = "../.tmp/images/animation_episode_{}_step_{}.png".format(self.episode_count, self.timestep)
                # filename = abs_path / filename
                # pygame.image.save(self.window_surface, filename)

        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.fps)

    def render_text_file(self, render_path):
        filename = abs_path / ('../.tmp/texts/' + render_path + '_episode_' + str(self.episode_count) + '_step_' + str(self.timestep) + '.txt')
        self.room = self.__get_room__()

        with open(filename, 'w') as f:
            # episode and timestep
            ep = 'Episode ' + str(self.episode_count)
            f.write(ep)
            ts = '\nTimestep ' + str(self.timestep) + '\n\n'
            f.write(ts)

            # upper separators
            separators = ''
            for _ in range(self.N):
                separators += ' -'
            f.write(separators)
            f.write('\n')

            # room content
            for row in range(self.room.shape[0]):
                f.write('|')
                for column in range(self.room.shape[1]):
                    element = self.room[row][column]
                    if element == self.agent_label:
                        f.write('A')
                    elif element == self.wardrobe_label:
                        f.write('W')
                    elif element == self.poison_label:
                        f.write('P')
                    elif element == self.treasure_label:
                        f.write('T')
                    else:
                        f.write('.')
                    f.write('|')
                f.write('\n')

            # lower separators
            f.write(separators)
            # f.write('\n')

    def close(self):
        pass

    def __init__room__(self):
        agent = wardrobe = poison = treasure = -1
        valid_init = False
        while not valid_init:
            agent, wardrobe, poison, treasure = np.random.choice(self.size, 4)
            if agent != wardrobe and wardrobe != poison and wardrobe != treasure and poison != treasure:
                valid_init = True

        return agent, wardrobe, poison, treasure

    def __init_agent_position__(self):
        agent = -1
        valid_init = False
        while not valid_init:
            agent = np.random.choice(self.size)
            if agent != self.wardrobe_pos:
                valid_init = True

        return agent

    def __get_room__(self):
        room = np.full(self.size, fill_value=-1)
        room[self.wardrobe_pos] = self.wardrobe_label
        room[self.poison_pos] = self.poison_label
        room[self.treasure_pos] = self.treasure_label
        room[self.agent_pos] = self.agent_label
        room = np.reshape(room, (self.N, self.N))

        return room

    def __get_encoded_room__(self):

        # no room change, we just encode the agent's position
        room = np.zeros(self.size)
        room[self.agent_pos] = 1

        # if the room disposition changes, we need to encode the rest of the objects
        if self.change_room:
            _room = np.zeros(shape=(self.size, 5))

            for i in range(_room.shape[0]):
                if i == self.agent_pos:
                    _room[i][self.agent_label + 1] = 1
                if i == self.wardrobe_pos:
                    _room[i][self.wardrobe_label + 1] = 1
                if i == self.poison_pos:
                    _room[i][self.poison_label + 1] = 1
                if i == self.treasure_pos:
                    _room[i][self.treasure_label + 1] = 1
                if np.sum(_room[i]) == 0:
                    _room[i][0] = 1

            return _room

        return room

    def print_separators(self):
        for _ in range(self.N):
            print(' -', end='')

    def print_element(self, element):
        if element == self.agent_label:
            print('A', end='')
        elif element == self.wardrobe_label:
            print('W', end='')
        elif element == self.poison_label:
            print('P', end='')
        elif element == self.treasure_label:
            print('T', end='')
        else:
            print('.', end='')
        print('|', end='')

    def is_adj(self, action):
        # if the agent is located in the center part of the room, he/she can move anywhere
        # if the agent is in the first row (upper limit), he/she cannot move up
        # if the agent is in the first column (left limit), he/she cannot move left
        # if the agent is in the last column (right limit), he/she cannot move right
        # if the agent is in the last row (lower limit), he/she cannot move down

        upper_limit = left_limit = right_limit = lower_limit = False

        if self.agent_pos in list(range(self.N)):  # agent is in the first row
            upper_limit = True
        if self.agent_pos % self.N == 0:  # agent is in the first column
            left_limit = True
        if self.agent_pos in list(range(self.N - 1, self.size, self.N)):  # agent is in the last column
            right_limit = True
        if self.agent_pos in list(range(self.size - self.N, self.size, 1)):  # agent is in the last row
            lower_limit = True

        if action == 0 and upper_limit:
            if self.verbose == 1:
                print(f'cannot move up: non adjacent block')
            return False
        if action == 1 and left_limit:
            if self.verbose == 1:
                print(f'cannot move left: non adjacent block')
            return False
        if action == 2 and right_limit:
            if self.verbose == 1:
                print(f'cannot move right: non adjacent block')
            return False
        if action == 3 and lower_limit:
            if self.verbose == 1:
                print(f'cannot move down: non adjacent block')
            return False
        return True

    def is_wardrobe(self, action):
        if action == 0 and self.agent_pos - self.N == self.wardrobe_pos:
            if self.verbose == 1:
                print(f'cannot move up: wardrobe blocking the way')
            return True
        elif action == 1 and self.agent_pos - 1 == self.wardrobe_pos:
            if self.verbose == 1:
                print(f'cannot move left: wardrobe blocking the way')
            return True
        elif action == 2 and self.agent_pos + 1 == self.wardrobe_pos:
            if self.verbose == 1:
                print(f'cannot move right: wardrobe blocking the way')
            return True
        elif action == 3 and self.agent_pos + self.N == self.wardrobe_pos:
            if self.verbose == 1:
                print(f'cannot move down: wardrobe blocking the way')
            return True
        return False
