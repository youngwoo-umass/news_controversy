
class flags:
    def __init__(self):
        self.batch_size_sm = 16
        self.batch_size_lg = 15
        self.log_device_placement = False
        self.comment_count = 100
        self.comment_length = 50
        self.embedding_size = 100
        self.article_length = 1000

FLAGS = flags()
