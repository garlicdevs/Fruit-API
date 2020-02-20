import cocos

from mazeexp.engine import config


class ScoreLayer(cocos.layer.Layer):

    """
    ScoreLayer

    Score overlay

    Responsabilities:
        display score and battery
    """
    def __init__(self, stats=None):
        super(ScoreLayer, self).__init__()

        self.stats = stats

        self.labels = {
            "score": "Score: ",
            "battery": "Battery: "
        }

        w, h = config.settings['window']['width'], config.settings['window']['height']
        lineheight = 15
        offset_x = -lineheight
        offset_y = lineheight

        self.msgs = {}

        for key in self.labels:
            value = self.labels[key]
            str_val = str(int(self.stats[key]))
            msg = cocos.text.Label(self.labels[key] + str_val,
                                    bold=True,
                                    color=(255, 50, 0, 255),
                                    font_size=12,
                                    font_name=config.settings['view']['font_name'],
                                    anchor_y='top',
                                    anchor_x='right',
                                    width=w,
                                    multiline=True,
                                    align="right")
            msg.position = (w+offset_x, h-offset_y)

            shad = cocos.text.Label(self.labels[key] + str_val,
                                    bold=True,
                                    color=(0, 0, 0, 255),
                                    font_size=12,
                                    font_name=config.settings['view']['font_name'],
                                    anchor_y='top',
                                    anchor_x='right',
                                    width=w,
                                    multiline=True,
                                    align="right")
            shad.scale_y *= 1.02
            shad.position = (w+offset_x+1, h-offset_y-1)
            self.msgs[key] = msg, shad
            self.add(shad)
            self.add(msg)

            offset_y += lineheight

        self.schedule(self.update)

    def update(self, dt):
        """
        Responsabilities:
            Updates game engine each tick
            Copies new stats into labels
        """
        for key in self.labels:
            str_val = str(int(self.stats[key]))
            self.msgs[key][0].element.text = self.labels[key] + str_val
            self.msgs[key][1].element.text = self.labels[key] + str_val
