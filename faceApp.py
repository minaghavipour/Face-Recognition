"""this is the main script of face verification (authentication/authorization) service.
jobs:
1. extracts features of a database and saves them in monster.
2. does authentication or authorization process.
"""

from os.path import join
from numpy import array
from cv2 import imread
from os import listdir
from tqdm import tqdm
import codecs
import pickle
import glob
import yaml
import sys

from verification.database import MyMonster
from verification import FaceVerifier


class faceEngine():
    """This class makes a database of faces, or does authentication or autorization.

    Attributes:
        verifier (face_verification.FaceVerifier): It finds name / feature of the closest face in an image.
        monster (monster.MyMonster): The endpoint, pointing to the monster docker container.
        database (str): A local path where images of different persons are saved.
        container (str): A container in monster where features are saved.
        object (str): An object in container (in monster) representing the features of faces.

    """
    def __init__(self):

        with open("config.yaml", 'r') as f:
            cfg = yaml.safe_load(f)

        self.verifier = FaceVerifier(cfg)
        self.monster = MyMonster(cfg["endpoint"])
        self.database = cfg["database"]
        self.container = cfg["container"] 
        self.object = cfg["object"]

    def init_feats(self):
        """This method is called once at the beginning of api run.

        """
        try:
            # get features from mosnster (path at monster: container/object)
            content = self.monster.getObjectInfo(self.container, self.object)
            # decode features from base64 to dictionary object (so feats is a dictionary)
            feats = pickle.loads(codecs.decode(content.encode(), "base64"))
            # save features as an atribute in verifier object
            self.verifier.features = array(list(feats.values())).squeeze()
            # save names as an attribute in verifier object
            self.verifier.names = list(feats.keys())
            print("features are loaded from monster successfully")
        except:
            print("error in loading featrues from monster")
            sys.exit()

    def make_database(self):
        """This method is called when faceApp.py runs,
        it extracts features of all images in database and save them with their names in monster.

        """
        img_dir = join(self.database, "images")
        names = listdir(img_dir)
        feats = {}
        for name in tqdm(names):
            img_pths = sorted(glob.glob(join(img_dir, name, '*')))
            for i, pth in enumerate(img_pths):
                img = imread(pth)[:, :, ::-1]
                feats[name+'_'+str(i)] = self.verifier.extract_feat(img)
        # encode feats as a base64 object
        content = codecs.encode(pickle.dumps(feats), "base64").decode()
        # create a container in monster
        self.monster.createContainer(self.container)
        # crreate an object in container with the content of feats
        self.monster.createObject(self.container, self.object, content)

    def authenticate(self, img):
        """This method is called when using Authentication service.

        Args:
            img (numpy.ndarray): Input image of service.

        """
        authenticated, name, conf = self.verifier.verify_one(img)
        result = {"authenticated": authenticated, "name": name,
                  "max_confidence": round(conf.astype("float64"), 3)}

        return result

    def authorize(self, img, uname):
        """This method is called when using Authorization service.

        Args:
            img (numpy.ndarray): Input image of service.
            uname (str): Name of user to be authorized.

        """
        authenticated, name, conf = self.verifier.verify_one(img)
        authorized = name.lower() == uname.lower() and authenticated
        result = {"authorized": authorized, "name": name,
                  "max_confidence": round(conf.astype("float64"), 3)}

        return result


if __name__ == "__main__":
    engine = faceEngine()
    engine.make_database()
