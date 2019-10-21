from fruit.envs.games.tank_battle.constants import GlobalConstants
from fruit.envs.games.tank_battle.manager import ResourceManager
from fruit.envs.games.tank_battle.map import ConditionalMap
from fruit.envs.games.tank_battle.stages import StageMap
from fruit.envs.games.tank_battle.sprites import TankSprite, BulletSprite, WallSprite, BaseSprite, ExplosionSprite
from fruit.envs.games.utils import Utils
import numpy as np
import pygame
import os
import sys
import collections as cl
import matplotlib.pyplot as plt


class TankBattle(object):

    def __init__(self, render=False, speed=60, max_frames=100000, frame_skip=1,
                 seed=None, num_of_enemies=5, two_players=True, player1_human_control=True,
                 player2_human_control=False, debug=False, using_map=False, multi_target=True, strategy=0,
                 enable_terminal_one_player_die=False, use_2_maps=False, first_player=None, second_player=None):

        # Prepare internal data
        self.screen_size = GlobalConstants.SCREEN_SIZE
        self.tile_size = GlobalConstants.TILE_SIZE
        self.max_frames = max_frames
        self.rd = render
        self.screen = None
        self.speed = speed
        self.num_of_enemies = num_of_enemies
        self.sprites = pygame.sprite.Group()
        self.enemies = pygame.sprite.Group()
        self.players = pygame.sprite.Group()
        self.bullets_player = pygame.sprite.Group()
        self.bullets_enemy = pygame.sprite.Group()
        self.bases = pygame.sprite.Group()
        self.walls = pygame.sprite.Group()
        self.booms = pygame.sprite.Group()
        self.num_of_actions = GlobalConstants.NUM_OF_ACTIONS
        self.num_of_tiles = int(self.screen_size/self.tile_size)
        self.end_of_game = False
        self.is_debug = debug
        self.frames_count = 0
        self.total_score = 0
        self.total_score_p1 = 0
        self.total_score_p2 = 0
        self.enemy_update_freq = 1
        self.bullet_speed = GlobalConstants.BULLET_SPEED
        self.font_size = GlobalConstants.FONT_SIZE
        self.player1_human_control = player1_human_control
        self.player2_human_control = player2_human_control
        self.two_players = two_players
        self.log_freq = 60
        if self.log_freq == 0:
            self.log_freq = 60
        self.current_stage = 0
        self.current_path = os.path.dirname(os.path.abspath(__file__))
        self.player_speed = GlobalConstants.PLAYER_SPEED
        self.enemy_speed = GlobalConstants.ENEMY_SPEED
        self.enemy_bullet_loading_time = GlobalConstants.ENEMY_LOADING_TIME
        self.current_buffer = np.array([[[0, 0, 0] for _ in range(self.screen_size)] for _ in range(self.screen_size)])
        self.pareto_solutions = None
        self.frame_speed = 0
        self.frame_skip = frame_skip
        self.started_time = Utils.get_current_time()
        self.next_rewards_p1 = cl.deque(maxlen=100)
        self.next_rewards_p2 = cl.deque(maxlen=100)
        self.map_rewards = cl.deque(maxlen=100)
        self.num_of_objs = 2
        self.using_map = using_map
        self.use_2_maps = use_2_maps
        self.multi_target = multi_target
        if self.using_map:
            self.num_of_objs = 3
        self.strategy = strategy
        self.enable_terminal = enable_terminal_one_player_die
        self.first_player = first_player
        self.second_player = second_player
        self.steps = 0

        if self.player1_human_control or self.player2_human_control:
            if not self.rd:
                raise ValueError("Invalid parameter ! Human control must be in rendering mode")

        # Seed is used to generate a stochastic environment
        if seed is None or seed < 0 or seed >= 9999:
            self.seed = np.random.randint(0, 9999)
            self.random_seed = True
        else:
            self.random_seed = False
            self.seed = seed
            np.random.seed(seed)

        # Initialize
        self.__init_pygame_engine()

        # Create base and walls
        self.__generate_base_and_walls()

        # Create players
        self.__generate_players()

        # Create enemies
        self.__generate_enemies(self.num_of_enemies)

        # Load map
        self.stage_map.load_map(self.current_stage)

        # Render the first frame
        self.__render()

    @staticmethod
    def get_game_name():
        return "TANK BATTLE"

    def clone(self):
        if self.random_seed:
            seed = np.random.randint(0, 9999)
        else:
            seed = self.seed
        return TankBattle(render=self.rd, speed=self.speed, max_frames=self.max_frames, frame_skip=self.frame_skip,
                          seed=seed, num_of_enemies=self.num_of_enemies, two_players=self.two_players,
                          player1_human_control=self.player1_human_control,
                          player2_human_control=self.player2_human_control,
                          debug=self.is_debug, using_map=self.using_map, multi_target=self.multi_target,
                          strategy=self.strategy,
                          enable_terminal_one_player_die=self.enable_terminal,
                          use_2_maps=self.use_2_maps, first_player=self.first_player, second_player=self.second_player
                          )

    def get_num_of_objectives(self):
        return self.num_of_objs

    def get_seed(self):
        return self.seed

    def __init_pygame_engine(self):
        # Center the screen
        os.environ['SDL_VIDEO_CENTERED'] = '1'

        # Init Pygame engine
        pygame.init()

        # Init joysticks
        self.num_of_joysticks = pygame.joystick.get_count()
        self.joystick_p1 = None
        self.joystick_p2 = None
        if self.num_of_joysticks > 0:
            self.joystick_p1 = pygame.joystick.Joystick(0)
            self.joystick_p1.init()
        if self.num_of_joysticks > 1:
            self.joystick_p2 = pygame.joystick.Joystick(1)
            self.joystick_p2.init()

        if self.rd:
            pygame.display.set_caption(TankBattle.get_game_name())
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        else:
            self.screen = pygame.Surface((self.screen_size, self.screen_size))
        self.rc_manager = ResourceManager(current_path=self.current_path, font_size=self.font_size,
                                          tile_size=self.tile_size, is_render=self.rd)
        self.font = self.rc_manager.get_font()
        self.stage_map = StageMap(self.num_of_tiles, tile_size=self.tile_size, current_path=self.current_path,
                                  sprites=self.sprites, walls=self.walls, resources_manager=self.rc_manager)
        self.map = ConditionalMap(self.screen_size, self.num_of_tiles, self.enemies, self.multi_target, self.strategy)
        self.map_buffer = np.array([[[0, 0, 0] for _ in range(self.screen_size)] for _ in range(self.screen_size)])

        if self.use_2_maps:
            self.map_2 = ConditionalMap(self.screen_size, self.num_of_tiles, self.enemies, False, 2)
            self.map_buffer_2 = np.array([[[0, 0, 0] for _ in range(self.screen_size)] for _ in range(self.screen_size)])

    def __generate_base_and_walls(self):
        # Create a base
        self.base = BaseSprite(self.tile_size, pos_x=int(self.num_of_tiles / 2), pos_y=self.num_of_tiles - 2,
                               sprite_bg=self.rc_manager.get_image(ResourceManager.BASE))
        self.sprites.add(self.base)
        self.bases.add(self.base)

        # Create walls
        wall_bg = self.rc_manager.get_image(ResourceManager.HARD_WALL)
        for i in range(self.num_of_tiles):
            wall_top = WallSprite(self.tile_size, i, 0, wall_bg)
            self.sprites.add(wall_top)
            self.walls.add(wall_top)

            wall_bottom = WallSprite(self.tile_size, i, self.num_of_tiles-1, wall_bg)
            self.sprites.add(wall_bottom)
            self.walls.add(wall_bottom)

            wall_left = WallSprite(self.tile_size, 0, i, wall_bg)
            self.sprites.add(wall_left)
            self.walls.add(wall_left)

            wall_right = WallSprite(self.tile_size, self.num_of_tiles-1, i, wall_bg)
            self.sprites.add(wall_right)
            self.walls.add(wall_right)

    def __generate_players(self):
        pos_x = int(self.num_of_tiles / 2) - 2
        pos_y = self.num_of_tiles - 2
        if self.first_player is not None:
            pos_x = self.first_player[0]
            pos_y = self.first_player[1]
        self.player1 = TankSprite(self.tile_size, pos_x=pos_x, pos_y=pos_y,
                                  sprite_bg=(self.rc_manager.get_image(ResourceManager.PLAYER1_LEFT),
                                             self.rc_manager.get_image(ResourceManager.PLAYER1_RIGHT),
                                             self.rc_manager.get_image(ResourceManager.PLAYER1_UP),
                                             self.rc_manager.get_image(ResourceManager.PLAYER1_DOWN)),
                                  is_enemy=False, bullet_loading_time=GlobalConstants.PLAYER_LOADING_TIME,
                                  speed=self.player_speed,
                                  auto_control=self.player1_human_control)
        self.sprites.add(self.player1)
        self.players.add(self.player1)

        if self.two_players:
            pos_x = int(self.num_of_tiles / 2) + 2
            pos_y = self.num_of_tiles - 2
            if self.second_player is not None:
                pos_x = self.second_player[0]
                pos_y = self.second_player[1]
            self.player2 = TankSprite(self.tile_size, pos_x=pos_x, pos_y=pos_y,
                                      sprite_bg=(self.rc_manager.get_image(ResourceManager.PLAYER2_LEFT),
                                                 self.rc_manager.get_image(ResourceManager.PLAYER2_RIGHT),
                                                 self.rc_manager.get_image(ResourceManager.PLAYER2_UP),
                                                 self.rc_manager.get_image(ResourceManager.PLAYER2_DOWN)),
                                      is_enemy=False, bullet_loading_time=GlobalConstants.PLAYER_LOADING_TIME,
                                      speed=self.player_speed, auto_control=True)
            self.sprites.add(self.player2)
            self.players.add(self.player2)

    def __generate_enemies(self, num_of_enemies):
        for _ in range(num_of_enemies):
            x = np.random.randint(1, self.num_of_tiles-1)
            y = np.random.randint(1, int(self.num_of_tiles / 2)-1)
            enemy = TankSprite(self.tile_size, pos_x=x, pos_y=y,
                               sprite_bg=(self.rc_manager.get_image(ResourceManager.ENEMY_LEFT),
                                          self.rc_manager.get_image(ResourceManager.ENEMY_RIGHT),
                                          self.rc_manager.get_image(ResourceManager.ENEMY_UP),
                                          self.rc_manager.get_image(ResourceManager.ENEMY_DOWN)),
                               is_enemy=True, bullet_loading_time=self.enemy_bullet_loading_time,
                               speed=self.enemy_speed,
                               auto_control=True)
            self.sprites.add(enemy)
            self.enemies.add(enemy)

            # Increase difficulty
            if self.total_score > 200:
                self.enemy_bullet_loading_time = GlobalConstants.ENEMY_LOADING_TIME - 10
            elif self.total_score > 500:
                self.enemy_bullet_loading_time = GlobalConstants.ENEMY_LOADING_TIME - 20
            elif self.total_score > 1000:
                self.enemy_speed = GlobalConstants.PLAYER_SPEED
                self.enemy_bullet_loading_time = GlobalConstants.ENEMY_LOADING_TIME - 20

    def __enemies_update(self):
        if self.frames_count % self.enemy_update_freq == 0:
            for enemy in self.enemies:
                rand_action = np.random.randint(0, self.num_of_actions)
                if rand_action != GlobalConstants.FIRE_ACTION:
                    rand_action = enemy.direction
                    if not enemy.move(rand_action, self.sprites):
                        rand_action = np.random.randint(0, self.num_of_actions)
                        if rand_action != GlobalConstants.FIRE_ACTION:
                            enemy.move(rand_action, self.sprites)
                        else:
                            enemy.fire_started_time = self.frames_count
                else:
                    self.__fire_bullet(enemy, True)

    def __draw_score(self):
        total_score = self.font.render('Score:' + str(self.total_score), False, Utils.get_color(Utils.WHITE))
        self.screen.blit(total_score, (self.screen_size/2 - total_score.get_width()/2,
                                       self.screen_size-self.tile_size + total_score.get_height()/1.3))

        p1_score = self.font.render('P1:' + str(self.total_score_p1), False, Utils.get_color(Utils.WHITE))
        self.screen.blit(p1_score, (10, self.screen_size-self.tile_size + p1_score.get_height()/1.3))

        p2_score = self.font.render('P2:' + str(self.total_score_p2), False, Utils.get_color(Utils.WHITE))
        self.screen.blit(p2_score, (self.screen_size - p2_score.get_width() - 10,
                                    self.screen_size-self.tile_size + p2_score.get_height()/1.3))

        stage_text = self.font.render('Stage ' + str(self.current_stage + 1), False, Utils.get_color(Utils.WHITE))
        self.screen.blit(stage_text, (self.screen_size/2 - stage_text.get_width()/2, stage_text.get_height()/1.3))

    def __fire_bullet(self, tank, is_enemy):
        if tank.is_terminate:
            return
        current_time = self.frames_count
        if tank is self.player1:
            owner = GlobalConstants.PLAYER_1_OWNER
        elif self.two_players and tank is self.player2:
            owner = GlobalConstants.PLAYER_2_OWNER
        else:
            owner = GlobalConstants.ENEMY_OWNER
        if current_time - tank.fire_started_time > tank.loading_time:
            tank.fire_started_time = self.frames_count
            bullet = BulletSprite(size=self.rc_manager.bullet_size,
                                  tile_size=self.tile_size,
                                  direction=tank.direction,
                                  speed=self.bullet_speed,
                                  pos_x=tank.target_x,
                                  pos_y=tank.target_y,
                                  owner=owner,
                                  sprite_bg=self.rc_manager.get_image(ResourceManager.BULLET))
            if is_enemy:
                self.bullets_enemy.add(bullet)
            else:
                self.bullets_player.add(bullet)
            self.sprites.add(bullet)

    @staticmethod
    def __is_key_pressed():
        keys = pygame.key.get_pressed()
        for i in range(len(keys)):
            if keys[i] != 0:
                return i
        return -1

    def __joystick_control(self):
        if self.player1_human_control and self.player2_human_control:
            if self.two_players:
                if self.joystick_p1 is not None:
                    print(self.joystick_p1.get_axis(0), self.joystick_p1.get_axis(1), self.joystick_p1.get_button(0), self.joystick_p1.get_button(1))
                    print(self.joystick_p1.get_button(2),
                          self.joystick_p1.get_button(3))
                    print("---")
                    if self.joystick_p1.get_axis(0) < -0.5:
                        self.player1.move(GlobalConstants.LEFT_ACTION, self.sprites)
                    if self.joystick_p1.get_axis(0) > 0.5:
                        self.player1.move(GlobalConstants.RIGHT_ACTION, self.sprites)
                    if self.joystick_p1.get_axis(1) < -0.5:
                        self.player1.move(GlobalConstants.UP_ACTION, self.sprites)
                    if self.joystick_p1.get_axis(1) > 0.5:
                        self.player1.move(GlobalConstants.DOWN_ACTION, self.sprites)
                    if self.joystick_p1.get_button(0) > 0.5 or self.joystick_p1.get_button(1) > 0.5:
                        self.__fire_bullet(self.player1, False)
                if self.joystick_p2 is not None:
                    if self.joystick_p2.get_axis(0) < -0.5:
                        self.player2.move(GlobalConstants.LEFT_ACTION, self.sprites)
                    if self.joystick_p2.get_axis(0) > 0.5:
                        self.player2.move(GlobalConstants.RIGHT_ACTION, self.sprites)
                    if self.joystick_p2.get_axis(1) < -0.5:
                        self.player2.move(GlobalConstants.UP_ACTION, self.sprites)
                    if self.joystick_p2.get_axis(1) > 0.5:
                        self.player2.move(GlobalConstants.DOWN_ACTION, self.sprites)
                    if self.joystick_p2.get_button(0) > 0.5 or self.joystick_p2.get_button(1) > 0.5:
                        self.__fire_bullet(self.player2, False)
            else:
                if self.joystick_p1 is not None:
                    if self.joystick_p1.get_axis(0) < -0.5:
                        self.player1.move(GlobalConstants.LEFT_ACTION, self.sprites)
                    if self.joystick_p1.get_axis(0) > 0.5:
                        self.player1.move(GlobalConstants.RIGHT_ACTION, self.sprites)
                    if self.joystick_p1.get_axis(1) < -0.5:
                        self.player1.move(GlobalConstants.UP_ACTION, self.sprites)
                    if self.joystick_p1.get_axis(1) > 0.5:
                        self.player1.move(GlobalConstants.DOWN_ACTION, self.sprites)
                    if self.joystick_p1.get_button(0) > 0.5 or self.joystick_p1.get_button(1) > 0.5:
                        self.__fire_bullet(self.player1, False)
        else:
            if not self.player1_human_control:
                if self.two_players:
                    if self.joystick_p2 is not None:
                        if self.joystick_p2.get_axis(0) < -0.5:
                            self.player2.move(GlobalConstants.LEFT_ACTION, self.sprites)
                        if self.joystick_p2.get_axis(0) > 0.5:
                            self.player2.move(GlobalConstants.RIGHT_ACTION, self.sprites)
                        if self.joystick_p2.get_axis(1) < -0.5:
                            self.player2.move(GlobalConstants.UP_ACTION, self.sprites)
                        if self.joystick_p2.get_axis(1) > 0.5:
                            self.player2.move(GlobalConstants.DOWN_ACTION, self.sprites)
                        if self.joystick_p2.get_button(0) > 0.5 or self.joystick_p2.get_button(1) > 0.5:
                            self.__fire_bullet(self.player2, False)
            else:
                if self.joystick_p1 is not None:
                    if self.joystick_p1.get_axis(0) < -0.5:
                        self.player1.move(GlobalConstants.LEFT_ACTION, self.sprites)
                    if self.joystick_p1.get_axis(0) > 0.5:
                        self.player1.move(GlobalConstants.RIGHT_ACTION, self.sprites)
                    if self.joystick_p1.get_axis(1) < -0.5:
                        self.player1.move(GlobalConstants.UP_ACTION, self.sprites)
                    if self.joystick_p1.get_axis(1) > 0.5:
                        self.player1.move(GlobalConstants.DOWN_ACTION, self.sprites)
                    if self.joystick_p1.get_button(0) > 0.5 or self.joystick_p1.get_button(1) > 0.5:
                        self.__fire_bullet(self.player1, False)

    def __human_control(self, key):
        if self.player1_human_control and self.player2_human_control:
            if self.two_players:
                if key == pygame.K_LEFT:
                    self.player1.move(GlobalConstants.LEFT_ACTION, self.sprites)
                if key == pygame.K_RIGHT:
                    self.player1.move(GlobalConstants.RIGHT_ACTION, self.sprites)
                if key == pygame.K_UP:
                    self.player1.move(GlobalConstants.UP_ACTION, self.sprites)
                if key == pygame.K_DOWN:
                    self.player1.move(GlobalConstants.DOWN_ACTION, self.sprites)
                if key == pygame.K_KP_ENTER:
                    self.__fire_bullet(self.player1, False)
                if key == pygame.K_a:
                    self.player2.move(GlobalConstants.LEFT_ACTION, self.sprites)
                if key == pygame.K_d:
                    self.player2.move(GlobalConstants.RIGHT_ACTION, self.sprites)
                if key == pygame.K_w:
                    self.player2.move(GlobalConstants.UP_ACTION, self.sprites)
                if key == pygame.K_s:
                    self.player2.move(GlobalConstants.DOWN_ACTION, self.sprites)
                if key == pygame.K_SPACE:
                    self.__fire_bullet(self.player2, False)
            else:
                if key == pygame.K_LEFT:
                    self.player1.move(GlobalConstants.LEFT_ACTION, self.sprites)
                if key == pygame.K_RIGHT:
                    self.player1.move(GlobalConstants.RIGHT_ACTION, self.sprites)
                if key == pygame.K_UP:
                    self.player1.move(GlobalConstants.UP_ACTION, self.sprites)
                if key == pygame.K_DOWN:
                    self.player1.move(GlobalConstants.DOWN_ACTION, self.sprites)
                if key == pygame.K_SPACE:
                    self.__fire_bullet(self.player1, False)
        else:
            if not self.player1_human_control:
                if self.two_players:
                    if key == pygame.K_LEFT:
                        self.player2.move(GlobalConstants.LEFT_ACTION, self.sprites)
                    if key == pygame.K_RIGHT:
                        self.player2.move(GlobalConstants.RIGHT_ACTION, self.sprites)
                    if key == pygame.K_UP:
                        self.player2.move(GlobalConstants.UP_ACTION, self.sprites)
                    if key == pygame.K_DOWN:
                        self.player2.move(GlobalConstants.DOWN_ACTION, self.sprites)
                    if key == pygame.K_SPACE:
                        self.__fire_bullet(self.player2, False)
            else:
                if key == pygame.K_LEFT:
                    self.player1.move(GlobalConstants.LEFT_ACTION, self.sprites)
                if key == pygame.K_RIGHT:
                    self.player1.move(GlobalConstants.RIGHT_ACTION, self.sprites)
                if key == pygame.K_UP:
                    self.player1.move(GlobalConstants.UP_ACTION, self.sprites)
                if key == pygame.K_DOWN:
                    self.player1.move(GlobalConstants.DOWN_ACTION, self.sprites)
                if key == pygame.K_SPACE:
                    self.__fire_bullet(self.player1, False)

    def __handle_event(self):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.reset()
                sys.exit()

        if not self.player1_human_control and not self.player2_human_control:
            return True

        key = TankBattle.__is_key_pressed()
        if key >= 0:
            self.__human_control(key)

        if self.num_of_joysticks > 0:
            self.__joystick_control()

        return True

    def get_key_pressed(self):
        return TankBattle.__is_key_pressed()

    def __check_reward(self):
        p1_score = 0
        if len(self.next_rewards_p1) > 0:
            p1_score = self.next_rewards_p1[0]
            self.next_rewards_p1.popleft()
        p2_score = 0
        if len(self.next_rewards_p2) > 0:
            p2_score = self.next_rewards_p2[0]
            self.next_rewards_p2.popleft()
        if self.using_map:
            map_score = 0
            if len(self.map_rewards) > 0:
                map_score = self.map_rewards[0]
                self.map_rewards.popleft()
            return [p1_score, p2_score, map_score]
        else:
            return [p1_score, p2_score]

    def __generate_explosion(self, abs_x, abs_y):
        expl = ExplosionSprite(self.tile_size, abs_x, abs_y, 2,
                               [self.rc_manager.get_image(ResourceManager.EXPLOSION_1),
                                self.rc_manager.get_image(ResourceManager.EXPLOSION_2),
                                self.rc_manager.get_image(ResourceManager.EXPLOSION_3)])
        self.sprites.add(expl)
        self.booms.add(expl)

    def __remove_explosions(self):
        for expl in self.booms:
            if expl.done():
                self.sprites.remove(expl)
                self.booms.remove(expl)

    def __bullets_update(self):
        for bullet in self.bullets_player:
            is_hit = False

            # Check if it hits other enemy's bullets
            bullets_hit = pygame.sprite.spritecollide(bullet, self.bullets_enemy, True)
            for bullet_enemy in bullets_hit:
                self.bullets_enemy.remove(bullet_enemy)
                self.sprites.remove(bullet_enemy)
                self.sprites.remove(bullet)
                self.bullets_player.remove(bullet)
                is_hit = True
                break
            if is_hit:
                continue

            # Check if it hits the enemy
            enemies_hit = pygame.sprite.spritecollide(bullet, self.enemies, True)
            for enemy in enemies_hit:
                self.__generate_explosion(enemy.rect.x, enemy.rect.y)
                self.enemies.remove(enemy)
                self.sprites.remove(enemy)
                self.sprites.remove(bullet)
                self.bullets_player.remove(bullet)
                self.total_score = self.total_score + 10
                if self.using_map and enemy.is_in_map:
                    self.map_rewards.append(10)
                if bullet.owner == GlobalConstants.PLAYER_1_OWNER:
                    self.total_score_p1 = self.total_score_p1 + 10
                    self.next_rewards_p1.append(10)
                else:
                    self.total_score_p2 = self.total_score_p2 + 10
                    self.next_rewards_p2.append(10)
                self.__generate_enemies(1)
                is_hit = True
                break
            if is_hit:
                continue

            # Check if it hits the player
            players_hit = pygame.sprite.spritecollide(bullet, self.players, True)
            for player in players_hit:
                player.is_terminate = True
                self.__generate_explosion(player.rect.x, player.rect.y)
                self.players.remove(player)
                self.sprites.remove(player)
                self.sprites.remove(bullet)
                self.bullets_player.remove(bullet)
                is_hit = True
                break
            if is_hit:
                continue

            # Check if it hits the base
            bases_hit = pygame.sprite.spritecollide(bullet, self.bases, True)
            for base in bases_hit:
                self.__generate_explosion(base.rect.x, base.rect.y)
                self.bases.remove(base)
                self.sprites.remove(base)
                self.sprites.remove(bullet)
                self.bullets_player.remove(bullet)
                self.end_of_game = True
                return

            # Check if it hits the wall -> remove the bullet
            walls_hit = pygame.sprite.spritecollide(bullet, self.walls, False)
            for wall in walls_hit:
                if wall.type == GlobalConstants.SOFT_OBJECT:
                    self.sprites.remove(wall)
                    self.walls.remove(wall)
                if wall.type != GlobalConstants.TRANSPARENT_OBJECT:
                    self.sprites.remove(bullet)
                    self.bullets_player.remove(bullet)

        for bullet in self.bullets_enemy:
            is_hit = False

            # Check if it hits other player's bullets
            bullets_hit = pygame.sprite.spritecollide(bullet, self.bullets_player, True)
            for bullet_player in bullets_hit:
                self.bullets_player.remove(bullet_player)
                self.sprites.remove(bullet_player)
                self.sprites.remove(bullet)
                self.bullets_enemy.remove(bullet)
                is_hit = True
                break
            if is_hit:
                continue

            # Check if it hits the player
            players_hit = pygame.sprite.spritecollide(bullet, self.players, True)
            for player in players_hit:
                player.is_terminate = True
                self.__generate_explosion(player.rect.x, player.rect.y)
                self.players.remove(player)
                self.sprites.remove(player)
                self.sprites.remove(bullet)
                self.bullets_enemy.remove(bullet)
                is_hit = True
                break
            if is_hit:
                continue

            if len(self.players) == 0:
                self.end_of_game = True

            if self.enable_terminal and self.two_players and len(self.players) < 2:
                self.end_of_game = True

            # Check if it hits the base
            bases_hit = pygame.sprite.spritecollide(bullet, self.bases, True)
            for base in bases_hit:
                self.__generate_explosion(base.rect.x, base.rect.y)
                self.bases.remove(base)
                self.sprites.remove(base)
                self.sprites.remove(bullet)
                self.bullets_enemy.remove(bullet)
                self.end_of_game = True
                return

            # Check if it hits the wall -> remove the bullet
            walls_hit = pygame.sprite.spritecollide(bullet, self.walls, False)
            for wall in walls_hit:
                if wall.type == GlobalConstants.SOFT_OBJECT:
                    self.sprites.remove(wall)
                    self.walls.remove(wall)
                if wall.type != GlobalConstants.TRANSPARENT_OBJECT:
                    self.sprites.remove(bullet)
                    self.bullets_enemy.remove(bullet)

    def __calculate_fps(self):
        self.frames_count = self.frames_count + 1
        if self.max_frames > 0:
            if self.frames_count > self.max_frames:
                self.end_of_game = True
        current_time = Utils.get_current_time()
        if current_time > self.started_time:
            self.frame_speed = self.frames_count / (current_time - self.started_time)
        else:
            self.frame_speed = 0

    def __print_info(self):
        if self.is_debug:
            if self.frames_count % self.log_freq == 0:
                print("Number of players' bullets:", len(self.bullets_player))
                print("Number of enemies' bullets:", len(self.bullets_enemy))
                print("Current frame:", self.frames_count)
                print("Player 1 score:", self.total_score_p1)
                print("Player 2 score:", self.total_score_p2)
                print("Total score:", self.total_score)
                print("Number of players left", len(self.players))
                print("Frame speed (FPS):", self.frame_speed)
                print("")

    def __render(self):

        # Handle user event
        if self.rd:
            self.__handle_event()

        # Draw background first
        self.screen.fill(Utils.get_color(Utils.BLACK))

        # Update sprites
        self.sprites.update()

        # Update enemies
        self.__enemies_update()

        # Update map
        if self.using_map:
            self.map.get_state()
            if self.use_2_maps:
                self.map_2.get_state()

        # Update bullets
        self.__bullets_update()

        # Redraw all sprites
        self.sprites.draw(self.screen)

        # Draw score
        self.__draw_score()

        # Remove all explosions
        self.__remove_explosions()

        # Show to the screen what we're have drawn so far
        if self.rd:
            pygame.display.flip()

        # Maintain 20 fps
        pygame.time.Clock().tick(self.speed)

        # Calculate fps
        self.__calculate_fps()

        # Debug
        self.__print_info()

    def set_seed(self, seed):
        self.seed = seed

    def reset(self):
        global global_steps
        global_steps = 0

        self.end_of_game = False
        self.frames_count = 0
        self.enemy_speed = GlobalConstants.ENEMY_SPEED
        self.enemy_bullet_loading_time = GlobalConstants.ENEMY_LOADING_TIME
        self.started_time = Utils.get_current_time()

        for sprite in self.sprites:
            sprite.kill()

        self.__generate_base_and_walls()
        self.__generate_players()
        self.__generate_enemies(self.num_of_enemies)

        self.stage_map.load_map(self.current_stage)

        if self.is_debug:
            interval = Utils.get_current_time() - self.started_time
            print("#################  RESET GAME  ##################")
            print("Episode terminated after:", interval, "(s)")
            print("Total score:", self.total_score)
            print("Player 1 score:", self.total_score_p1)
            print("Player 2 score:", self.total_score_p2)
            print("#################################################")

        self.total_score = 0
        self.total_score_p1 = 0
        self.total_score_p2 = 0

        self.__render()

    def step(self, action, action_p2=-1):
        if self.player1_human_control and self.player2_human_control:
            raise ValueError("Error: human control mode")
        players = []
        if not self.player1_human_control and not self.player2_human_control:
            if self.two_players:
                if action == GlobalConstants.P1_LEFT_ACTION:
                    self.player1.move(GlobalConstants.LEFT_ACTION, self.sprites)
                elif action == GlobalConstants.P1_RIGHT_ACTION:
                    self.player1.move(GlobalConstants.RIGHT_ACTION, self.sprites)
                elif action == GlobalConstants.P1_UP_ACTION:
                    self.player1.move(GlobalConstants.UP_ACTION, self.sprites)
                elif action == GlobalConstants.P1_DOWN_ACTION:
                    self.player1.move(GlobalConstants.DOWN_ACTION, self.sprites)
                elif action == GlobalConstants.P1_FIRE_ACTION:
                    self.__fire_bullet(self.player1, False)

                if action_p2 == GlobalConstants.P2_LEFT_ACTION:
                    self.player2.move(GlobalConstants.LEFT_ACTION, self.sprites)
                elif action_p2 == GlobalConstants.P2_RIGHT_ACTION:
                    self.player2.move(GlobalConstants.RIGHT_ACTION, self.sprites)
                elif action_p2 == GlobalConstants.P2_UP_ACTION:
                    self.player2.move(GlobalConstants.UP_ACTION, self.sprites)
                elif action_p2 == GlobalConstants.P2_DOWN_ACTION:
                    self.player2.move(GlobalConstants.DOWN_ACTION, self.sprites)
                elif action_p2 == GlobalConstants.P2_FIRE_ACTION:
                    self.__fire_bullet(self.player2, False)
                players.append(GlobalConstants.PLAYER_1_OWNER)
                players.append(GlobalConstants.PLAYER_2_OWNER)
            else:
                if action != GlobalConstants.FIRE_ACTION:
                    self.player1.move(action, self.sprites)
                else:
                    self.__fire_bullet(self.player1, False)
                players.append(GlobalConstants.PLAYER_1_OWNER)
        else:
            if not self.player1_human_control:
                if action != GlobalConstants.FIRE_ACTION:
                    self.player1.move(action, self.sprites)
                else:
                    self.__fire_bullet(self.player1, False)
                players.append(GlobalConstants.PLAYER_1_OWNER)
            else:
                if self.two_players:
                    if action != GlobalConstants.FIRE_ACTION:
                        self.player2.move(action, self.sprites)
                    else:
                        self.__fire_bullet(self.player2, False)
                    players.append(GlobalConstants.PLAYER_2_OWNER)
                else:
                    raise ValueError("Error: human control mode")

        if self.frame_skip <= 1:
            self.__render()
        else:
            for _ in range(self.frame_skip):
                self.__render()

        return self.__check_reward()

    def render(self):
        self.__render()

    def step_all(self, action):
        r = self.step(action)
        next_state = self.get_state()
        terminal = self.is_terminal()
        return next_state, r, terminal

    def get_state_space(self):
        return [self.screen_size, self.screen_size]

    def get_action_space(self):
        return range(self.num_of_actions)

    def get_state(self):
        pygame.pixelcopy.surface_to_array(self.current_buffer, self.screen)
        return self.current_buffer

    def get_map(self, type=-1):
        if type == 1:
            map = self.map_2.get_state()
            pygame.pixelcopy.surface_to_array(self.map_buffer_2, map)
            return self.map_buffer_2
        else:
            map = self.map.get_state()
            pygame.pixelcopy.surface_to_array(self.map_buffer, map)
            return self.map_buffer

    def is_terminal(self):
        return self.end_of_game

    def debug(self):
        self.__print_info()

    def get_num_of_actions(self):
        return self.num_of_actions

    def is_render(self):
        return self.rd


def machine_control(two_players=False):
    import pickle
    game = TankBattle(render=False, player1_human_control=False, player2_human_control=False, two_players=two_players,
                      speed=2000, debug=False, frame_skip=5, using_map=False)
    scores = []
    scores_p1 = []
    scores_p2 = []
    num_of_actions = game.get_num_of_actions()
    game.reset()
    state = Utils.process_state(game.get_state())
    for step in range(100000):
        if two_players:
            random_action_p1 = np.random.randint(0, num_of_actions)
            random_action_p2 = np.random.randint(0, num_of_actions)
            reward = game.step(random_action_p1, random_action_p2)
        else:
            random_action = np.random.randint(0, num_of_actions)
            reward = game.step(random_action)

        is_terminal = game.is_terminal()

        if is_terminal:
            print("P1 Score:", game.total_score_p1)
            print("P2 Score:", game.total_score_p2)
            print("Total Score", game.total_score)
            print("Current steps:", step)
            scores.append(game.total_score)
            scores_p1.append(game.total_score_p1)
            scores_p2.append(game.total_score_p2)
            game.reset()

    if not two_players:
        with open('../examples/train/nips/tankbattle/Random/one_player/score_3.txt', 'wb') as file:
            pickle.dump(scores, file)

        with open('../examples/train/nips/tankbattle/Random/one_player/score_3.txt', 'rb') as file:
            load_scores = pickle.load(file)
            print(load_scores)
    else:
        with open('../examples/train/nips/tankbattle/Random/two_players/score_p1', 'wb') as file:
            pickle.dump(scores_p1, file)

        with open('../examples/train/nips/tankbattle/Random/two_players/score_p2', 'wb') as file:
            pickle.dump(scores_p2, file)

        with open('../examples/train/nips/tankbattle/Random/two_players/total_score', 'wb') as file:
            pickle.dump(scores, file)


def human_control(two_players=False):
    global global_steps
    import pickle

    game = TankBattle(render=True, player1_human_control=True, player2_human_control=True, two_players=two_players,
                      speed=80, debug=True, frame_skip=5)
    game.reset()
    scores = []
    steps = []
    scores_p1 = []
    scores_p2 = []
    total_steps = 0
    max_step = 1000*5
    for step in range(max_step):
        game.render()

        total_steps = total_steps + 1

        terminal = game.is_terminal()
        if terminal:
            print("P1 Score:", game.total_score_p1)
            print("P2 Score:", game.total_score_p2)
            print("Total Score", game.total_score)
            print("Current steps:", total_steps)
            scores.append(game.total_score)
            scores_p1.append(game.total_score_p1)
            scores_p2.append(game.total_score_p2)
            steps.append(total_steps/5)
            total_steps = 0
            game.reset()

    if not two_players:
        with open('../examples/train/nips/TankBattle/Human/one_player/score_3.txt', 'wb') as file:
            pickle.dump(scores, file)

        with open('../examples/train/nips/TankBattle/Human/one_player/step_3.txt', 'wb') as file:
            pickle.dump(steps, file)
            print(steps)

        with open('../examples/train/nips/TankBattle/Human/one_player/score_3.txt', 'rb') as file:
            load_scores = pickle.load(file)
            print(load_scores)
    else:
        with open('../examples/train/nips/TankBattle/Human/two_players/score_p1', 'wb') as file:
            pickle.dump(scores_p1, file)

        with open('../examples/train/nips/TankBattle/Human/two_players/score_p2', 'wb') as file:
            pickle.dump(scores_p2, file)

        with open('../examples/train/nips/TankBattle/Human/one_player/steps', 'wb') as file:
            pickle.dump(steps, file)

        with open('../examples/train/nips/TankBattle/Human/two_players/total_score', 'wb') as file:
            pickle.dump(scores, file)

        print(scores_p1)
        print(scores_p2)
        print(scores)
        print(np.mean(scores))
        print(steps)
        print(np.mean(steps))


def get_human_score_1_player():
    import pickle
    import matplotlib.pyplot as plt

    with open('../examples/train/nips/tankbattle/Human/one_player/score_2.txt', 'rb') as file:
        load_scores_3 = pickle.load(file)

    scores = load_scores_3[0:100]
    max_score = np.max(scores)
    mean_score = np.mean(scores)
    min_score = np.min(scores)
    print("Max Score:", max_score)
    print("Min score", min_score)
    print("Mean score:", mean_score)

    print(scores)
    plt.plot(range(1, len(scores)+1), [x for x in scores], '^')
    plt.savefig('../examples/train/nips/tankbattle/Human/one_player/reward_distribution.png')
    plt.show()


def get_human_score_2_players():
    import pickle
    import matplotlib.pyplot as plt

    with open('../examples/train/nips/tankbattle/Human/two_players/score.txt', 'rb') as file:
        load_scores_3 = pickle.load(file)

    scores = load_scores_3[0:100]
    max_score = np.max(scores)
    mean_score = np.mean(scores)
    min_score = np.min(scores)
    print("Max Score:", max_score)
    print("Min score", min_score)
    print("Mean score:", mean_score)

    print(scores)
    plt.plot(range(1, len(scores)+1), [x for x in scores], '^')
    plt.savefig('../examples/train/nips/tankbattle/Human/one_player/reward_distribution.png')
    plt.show()

    #180, 50, 100, 160, 120, 40, 330, 130, 170, 310, 40, 120, 190, 190, 180, 80, 70, 270, 230, 70, 160, 300, 150, 10, 40, 30, 90, 40, 290, 650, 210, 20, 110, 350, 770, 110, 20, 170, 440, 250, 150, 430, 390, 220, 190, 170, 210, 170, 200, 110, 200, 170, 60, 60, 100, 120, 300, 140, 150, 120, 280, 640, 210, 200, 170, 10, 130, 110, 240, 50, 120, 60, 390, 250, 10, 120, 130, 120, 150, 300, 70, 100, 70, 520, 370, 90, 150, 30, 80, 380, 230, 30, 100, 90, 0, 60, 70, 140, 260, 120]


def get_machine_score_1_player():
    import pickle

    with open('../examples/train/nips/tankbattle/Random/one_player/score_3.txt', 'rb') as file:
        load_scores_3 = pickle.load(file)

    scores = load_scores_3[0:100]
    max_score = np.max(scores)
    mean_score = np.mean(scores)
    min_score = np.min(scores)
    print("Max Score:", max_score)
    print("Min score", min_score)
    print("Mean score:", mean_score)

    print(scores)
    plt.plot(range(1, len(scores)+1), [x for x in scores], '^')
    plt.savefig('../examples/train/nips/tankbattle/Random/one_player/reward_distribution.png')
    plt.show()


def get_machine_score_2_players():
    import pickle

    with open('../examples/train/nips/tankbattle/Random/two_players/score_p1', 'rb') as file:
        load_scores_1 = pickle.load(file)

    with open('../examples/train/nips/tankbattle/Random/two_players/score_p2', 'rb') as file:
        load_scores_2 = pickle.load(file)

    with open('../examples/train/nips/tankbattle/Random/two_players/total_score', 'rb') as file:
        load_scores_3 = pickle.load(file)

    scores = load_scores_3
    max_score = np.max(scores)
    mean_score = np.mean(scores)
    min_score = np.min(scores)
    print("Max Score:", max_score)
    print("Min score", min_score)
    print("Mean score:", mean_score)

    print(scores)
    plt.plot(range(1, len(scores)+1), [x for x in scores], '^')
    plt.savefig('../examples/train/nips/tankbattle/Random/two_players/reward_distribution.png')
    plt.show()


def check_map():
    from PIL import Image
    from pathlib import Path
    home = str(Path.home())
    game = TankBattle(render=False, player1_human_control=False, player2_human_control=False, two_players=False,
                      speed=120, debug=True, frame_skip=5, using_map=True, num_of_enemies=5, multi_target=True,
                      strategy=3)
    num_of_actions = game.get_num_of_actions()
    game.reset()
    full_path = home + '/Desktop/Images/'
    count = 0

    while True:
        random_action = np.random.randint(0, num_of_actions)
        reward = game.step(random_action)
        print(reward)
        next_state = Utils.process_state(game.get_state())
        next_map = Utils.process_state(game.get_map())
        is_terminal = game.is_terminal()

        # save state and map
        count = count + 1
        img = Image.fromarray(next_state, 'L')
        img.save(full_path + str(count) +'.png')
        count = count + 1
        img = Image.fromarray(next_map, 'L')
        img.save(full_path + str(count) + '.png')

        if is_terminal:
            print("P1 Score:", game.total_score_p1)
            print("P2 Score:", game.total_score_p2)
            print("Total Score", game.total_score)
            game.reset()
            break


if __name__ == '__main__':

    # machine_control(True)

    # get_machine_score_1_player()

    # get_human_score_1_player()

    # get_machine_score_2_players()

    human_control(True)

    # get_human_score_1_player()

    # check_map()

    # human_control(two_players=True)

    # RESULT
    # Human 1 player: 118.2 - 142.7
    # Human 2 players: 175.5

    # Random 2 players: 7.16
    # Random 1 player: 5.2
