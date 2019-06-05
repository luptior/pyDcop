'''

'''

import threading
import mailer


def main():
    # Creating and starting the 'listen' thread
    listen = threading.Thread(name='mailer',
                              target=mailer.listen_func,
                              args=(msgs, listening_socket),
                              kwargs={'agent': agent})
    listen.setDaemon(True)
    listen.start()
