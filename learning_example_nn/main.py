#programm that makes a bunch of graphs to show how neural networks learn
#make them to a vid with Shutter Encoder

#own imports
from train import train_model
from graphics import make_img


#names for importing into ShutterEncoder for making videos
new_names = [
    "aa", "ab", "ac", "ad", "ae", "af", "ag", "ah", "ai", "aj", "ak", "al", "am", "an", "ao", "ap", "aq", "ar", "as", "at", "au", "av", "aw", "ax", "ay", "az",
    "ba", "bb", "bc", "bd", "be", "bf", "bg", "bh", "bi", "bj", "bk", "bl", "bm", "bn", "bo", "bp", "bq", "br", "bs", "bt", "bu", "bv", "bw", "bx", "by", "bz",
    "ca", "cb", "cc", "cd", "ce", "cf", "cg", "ch", "ci", "cj", "ck", "cl", "cm", "cn", "co", "cp", "cq", "cr", "cs", "ct", "cu", "cv", "cw", "cx", "cy", "cz"
]    

#make images
def main_img_gen(n_images=50, start_point=1, increment=1):
    cur_epoch = start_point
    for i in range(n_images):
        train_model(cur_epoch)
        make_img(new_names[i], epoch=cur_epoch)
        print(f"finished with epoch {cur_epoch}/{n_images}")
        cur_epoch += increment
        

#single generation of img
def test(epochs=50):
    train_model(epochs)
    make_img(epochs)

#let it run
#to change the training data function go to gen_learn_data line 10
main_img_gen()