
# %%

import logging

class DiLog:

    def __init__(self, _message_dict = None):
        self.message_dict = {}
        self.message_number  = 0
        if _message_dict is not None:
            self.message_dict = _message_dict

    def gen_message(self, _mess_id, _str):
        return '{}: {}'.format(_mess_id, _str)

    def out(self, _str):
        mess_id = 'm{:04}'.format(self.message_number)
        self.message_dict[mess_id] = _str
        logging.info(self.gen_message(mess_id, _str))
        self.message_number  += 1

    def get_html(self):
        str_html = ''
        for k, v in self.message_dict.items():
            str_html += self.gen_message(k, v) + '<br>'
        return str_html

dilog = DiLog()

def p(_str):
    global dilog
    dilog.out(_str)




