class Datasets:
    def __init__(self):
        super(Datasets, self).__init__()

    def get_dir(self):
        dir=''
        return dir

    @classmethod
    def attributed(cls):
        raise NotImplementedError

class ACM(Datasets):
    def __init__(self):
        super(ACM, self).__init__()

    def get_dir(self):
        dir='data/ACM/ACM.mat'
        return dir

    @classmethod
    def attributed(cls):
        return True


class Flickr(Datasets):
    def __init__(self):
        super(Flickr, self).__init__()

    def get_dir(self):
        dir = 'data/Flickr/Flickr_SDM.mat'
        return dir

    @classmethod
    def attributed(cls):
        return True

class BlogCatalog(Datasets):
    def __init__(self):
        super(BlogCatalog, self).__init__()

    def get_dir(self):
        dir = 'data/BlogCatalog/BlogCatalog.mat'
        return dir

    @classmethod
    def attributed(cls):
        return True
