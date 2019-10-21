from fruit.envs.games.utils import Utils
import pygame


class ConditionalMap:
    def __init__(self, size, num_of_tiles, enemies, multi_target, strategy):
        self.size = size
        self.enemies = enemies
        self.tile_size = int(size/num_of_tiles)
        self.num_of_tiles = num_of_tiles
        self.multi_target = multi_target
        self.strategy = strategy
        base_x = int(self.num_of_tiles / 2)
        base_y = self.num_of_tiles - 2
        self.check_spots = [[base_x - 1, base_y], [base_x + 1, base_y], [base_x - 2, base_y], [base_x + 2, base_y],
                            [base_x - 3, base_y], [base_x + 3, base_y], [base_x - 4, base_y], [base_x + 4, base_y],
                            [base_x - 5, base_y], [base_x + 5, base_y], [base_x - 5, base_y - 1],
                            [base_x + 5, base_y - 1],
                            [base_x - 5, base_y - 2], [base_x + 5, base_y - 2], [base_x - 5, base_y - 3],
                            [base_x + 5, base_y - 3],
                            [base_x - 5, base_y - 4], [base_x + 5, base_y - 4], [base_x, base_y - 1],
                            [base_x, base_y - 2],
                            [base_x, base_y - 3], [base_x, base_y - 4], [base_x, base_y - 5], [base_x, base_y - 6],
                            [base_x, base_y - 7], [base_x, base_y - 8], [base_x, base_y - 9], [base_x, base_y - 10],
                            [base_x - 5, base_y - 5], [base_x + 5, base_y - 5]]
        self.map = pygame.Surface((self.size, self.size))

    def get_state(self):
        self.map.fill(Utils.get_color(Utils.BLACK))
        if self.multi_target:
            if self.strategy == 0:
                for enemy in self.enemies:
                    if enemy.pos_y > 5 or enemy.pos_x == 6:
                        pygame.draw.rect(self.map, Utils.get_color(Utils.WHITE),
                                         (enemy.rect.x, enemy.rect.y, self.tile_size, self.tile_size))
                        enemy.is_in_map = True
                    else:
                        enemy.is_in_map = False
            elif self.strategy == 1:
                for enemy in self.enemies:
                    if (enemy.pos_y >= 6 and enemy.pos_y <= 8) or enemy.pos_x == 6:
                        pygame.draw.rect(self.map, Utils.get_color(Utils.WHITE),
                                         (enemy.rect.x, enemy.rect.y, self.tile_size, self.tile_size))
                        enemy.is_in_map = True
                    else:
                        enemy.is_in_map = False
            elif self.strategy == 2:
                for enemy in self.enemies:
                    if enemy.pos_y >= 9:
                        pygame.draw.rect(self.map, Utils.get_color(Utils.WHITE),
                                         (enemy.rect.x, enemy.rect.y, self.tile_size, self.tile_size))
                        enemy.is_in_map = True
                    else:
                        enemy.is_in_map = False
            elif self.strategy == 3:
                for enemy in self.enemies:
                    if enemy.pos_x == 6:
                        pygame.draw.rect(self.map, Utils.get_color(Utils.WHITE),
                                         (enemy.rect.x, enemy.rect.y, self.tile_size, self.tile_size))
                        enemy.is_in_map = True
                    else:
                        enemy.is_in_map = False
                for enemy in self.enemies:
                    if enemy.is_in_map and enemy.pos_y >= 6:
                        if enemy.pos_y <= 8 and enemy.pos_x > 6:
                            pygame.draw.rect(self.map, Utils.get_color(Utils.WHITE),
                                             (enemy.rect.x, enemy.rect.y, self.tile_size, self.tile_size))
                            return self.map
                        else:
                            enemy.is_in_map = False
                for enemy in self.enemies:
                    if enemy.pos_y >= 6 and enemy.pos_y <= 8 and enemy.pos_x > 6:
                        pygame.draw.rect(self.map, Utils.get_color(Utils.WHITE),
                                         (enemy.rect.x, enemy.rect.y, self.tile_size, self.tile_size))
                        enemy.is_in_map = True
                        return self.map
            elif self.strategy == 4:  # MSCS
                for enemy in self.enemies:
                    if enemy.pos_x == 6:
                        pygame.draw.rect(self.map, Utils.get_color(Utils.WHITE),
                                         (enemy.rect.x, enemy.rect.y, self.tile_size, self.tile_size))
                        enemy.is_in_map = True
                    else:
                        enemy.is_in_map = False
                for enemy in self.enemies:
                    if enemy.is_in_map and enemy.pos_y >= 6:
                        if enemy.pos_y <= 11:
                            pygame.draw.rect(self.map, Utils.get_color(Utils.WHITE),
                                             (enemy.rect.x, enemy.rect.y, self.tile_size, self.tile_size))
                            return self.map
                        else:
                            enemy.is_in_map = False
                for enemy in self.enemies:
                    if enemy.pos_y >= 6 and enemy.pos_y <= 11:
                        pygame.draw.rect(self.map, Utils.get_color(Utils.WHITE),
                                         (enemy.rect.x, enemy.rect.y, self.tile_size, self.tile_size))
                        enemy.is_in_map = True
                        return self.map
        else:
            if self.strategy == 0:  # Full defend
                for enemy in self.enemies:
                    if enemy.is_in_map:
                        if enemy.pos_y > 5 or enemy.pos_x == 6:
                            pygame.draw.rect(self.map, Utils.get_color(Utils.WHITE),
                                             (enemy.rect.x, enemy.rect.y, self.tile_size, self.tile_size))
                            return self.map
                        else:
                            enemy.is_in_map = False
                for enemy in self.enemies:
                    if enemy.pos_y > 5 or enemy.pos_x == 6:
                        pygame.draw.rect(self.map, Utils.get_color(Utils.WHITE),
                                            (enemy.rect.x, enemy.rect.y, self.tile_size, self.tile_size))
                        enemy.is_in_map = True
                        return self.map
            elif self.strategy == 1:  # Defend top
                for enemy in self.enemies:
                    if enemy.is_in_map:
                        if (enemy.pos_y >= 6 and enemy.pos_y <= 8) or enemy.pos_x == 6:
                            pygame.draw.rect(self.map, Utils.get_color(Utils.WHITE),
                                             (enemy.rect.x, enemy.rect.y, self.tile_size, self.tile_size))
                            return self.map
                        else:
                            enemy.is_in_map = False

                for enemy in self.enemies:
                    if (enemy.pos_y >= 6 and enemy.pos_y <= 8) or enemy.pos_x == 6:
                        pygame.draw.rect(self.map, Utils.get_color(Utils.WHITE),
                                         (enemy.rect.x, enemy.rect.y, self.tile_size, self.tile_size))
                        enemy.is_in_map = True
                        return self.map
            elif self.strategy == 2:  # Defend bottom
                for enemy in self.enemies:
                    if enemy.is_in_map:
                        if enemy.pos_y >= 9 and enemy.pos_x < 6:
                            pygame.draw.rect(self.map, Utils.get_color(Utils.WHITE),
                                             (enemy.rect.x, enemy.rect.y, self.tile_size, self.tile_size))
                            return self.map
                        else:
                            enemy.is_in_map = False

                for enemy in self.enemies:
                    if enemy.pos_y >= 9 and enemy.pos_x < 6:
                        pygame.draw.rect(self.map, Utils.get_color(Utils.WHITE),
                                         (enemy.rect.x, enemy.rect.y, self.tile_size, self.tile_size))
                        enemy.is_in_map = True
                        return self.map
            elif self.strategy == 3:  # Defend top left
                for enemy in self.enemies:
                    if enemy.is_in_map:
                        if (enemy.pos_y >= 6 and enemy.pos_y <= 8 and enemy.pos_x <= 6) or enemy.pos_x == 6:
                            pygame.draw.rect(self.map, Utils.get_color(Utils.WHITE),
                                             (enemy.rect.x, enemy.rect.y, self.tile_size, self.tile_size))
                            return self.map
                        else:
                            enemy.is_in_map = False

                for enemy in self.enemies:
                    if (enemy.pos_y >= 6 and enemy.pos_y <= 8 and enemy.pos_x <= 6) or enemy.pos_x == 6:
                        pygame.draw.rect(self.map, Utils.get_color(Utils.WHITE),
                                         (enemy.rect.x, enemy.rect.y, self.tile_size, self.tile_size))
                        enemy.is_in_map = True
                        return self.map
        return self.map