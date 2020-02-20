import cocos
import cocos.actions as ac

from mazeexp.engine import config


class MessageLayer(cocos.layer.Layer):

    """
    MessageLayer

    Transitory messages over WorldLayer

    Responsability:
    full display cycle for transitory messages, with effects and
    optional callback after hiding the message.
    """

    def show_message(self, msg, callback=None):
        w, h = config.settings['window']['width'], config.settings['window']['height']

        self.msg = cocos.text.Label(msg,
                                    font_size=52,
                                    font_name=config.settings['view']['font_name'],
                                    anchor_y='center',
                                    anchor_x='center',
                                    width=w,
                                    multiline=True,
                                    align="center")
        self.msg.position = (w / 2.0, h)

        self.add(self.msg)

        actions = (
            ac.Show() + ac.Accelerate(ac.MoveBy((0, -h / 2.0), duration=0.5)) +
            ac.Delay(1) +
            ac.Accelerate(ac.MoveBy((0, -h / 2.0), duration=0.5)) +
            ac.Hide()
        )

        if callback:
            actions += ac.CallFunc(callback)

        self.msg.do(actions)
