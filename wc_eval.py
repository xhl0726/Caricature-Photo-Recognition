'''
copied and modified from: https://github.com/clcarwin/sphereface_pytorch/blob/master/lfw_eval.py
'''
from __future__ import print_function
from torchvision import transforms
import argparse
import bisect
import datetime
import os
import pickle
import random
import sys
from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from matlab_cp2tform import get_similarity_transform_for_cv2
from net_sphere  import sphere20a

torch.backends.cudnn.bencmark = True


def calc_eye_point(landmark, is_right_eye=0):
        offset = is_right_eye * 2
        t = np.array([
            landmark[8+offset],
            landmark[9+offset],
        ])
        return t.mean(axis=0)


def get_img5point(landmark):
    return np.array([
        calc_eye_point(landmark, is_right_eye=0),       # Left eye
        calc_eye_point(landmark, is_right_eye=1),       # Right eye
        landmark[12],                                   # Nose tip
        landmark[14],                                   # Mouth left corner
        landmark[16],                                   # Mouth right corner
    ])


def load_landmark(file_path):
    return get_img5point([
        tuple(map(float, landmark.strip().split(' ')))\
        for landmark in\
        open(file_path).readlines()
    ])


def alignment(src_img, src_pts):
    ref_pts = [
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041],
    ]
    crop_size = (96, 112)
    src_pts = np.array(src_pts).reshape(5, 2)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    return face_img


def KFold(folds_length):
    folds = []
    n = sum(folds_length)
    n_folds = len(folds_length)
    l, r = 0, 0
    base = list(range(n))
    for i in range(n_folds):
        r += folds_length[i]
        test = base[l: r]
        train = list(set(base)-set(test))
        folds.append([train, test])
        l += folds_length[i]
    return folds


def eval_acc(threshold, diff):
    y_true = []
    y_predict = []
    for d in diff:
        same = 1 if float(d[0]) > threshold else 0
        y_predict.append(same)
        y_true.append(int(d[1]))
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    accuracy = 1.0*np.count_nonzero(y_true==y_predict)/len(y_true)
    return accuracy


def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold


def calc_roc(predicts):
    def get_tar(thd, pos_predicts):
        return 1 - bisect.bisect_left(pos_predicts, thd) / len(pos_predicts)
        # 在pos_predicts中查找thd,thd存在是返回thd左侧的位置，不存在返回应该插入的位置
    pos_predicts, neg_predicts = [], []
    for predict in predicts:
        if int(predict[1]) == 1:
            pos_predicts.append(float(predict[0]))
        else:
            neg_predicts.append(float(predict[0]))
    pos_predicts.sort()
    neg_predicts.sort()

    far3_idx = int(len(neg_predicts)-1-len(neg_predicts)*1e-3)
    far3 = get_tar(neg_predicts[far3_idx], pos_predicts)
    far2_idx = int(len(neg_predicts)-1-len(neg_predicts)*1e-2)
    far2 = get_tar(neg_predicts[far2_idx], pos_predicts)
    
    auc = 0
    n = len(pos_predicts) - 1
    for threshold in reversed(neg_predicts):
        while 0 <= n and threshold <= pos_predicts[n]:
            n -= 1
        auc += len(pos_predicts) - 1 - n
    auc /= len(pos_predicts) * len(neg_predicts)
    return far3, far2, auc


def resticted_fold_iter(dataset_path, folds_length):  # folds_length = []
    with open(os.path.join(
        dataset_path,
        'EvaluationProtocols',
        'FaceVerification',
        'Restricted',
        'RestrictedView2.txt',  # p_c
        # 'restricted_p_p.txt',
        # 'restricted_c_c.txt',
    )) as f:
        pairs_lines = iter(f.readlines())

    for fold_idx in range(10):
        fold_length = int(next(pairs_lines)) * 2  # 1580
        folds_length.append(fold_length)
        for i in range(fold_length):
            p = next(pairs_lines)
            p = p.replace('\n', '').split('\t')
            if i * 2 < fold_length:  # 注：在1-1580之间，前790个是同一个人的图片对，但是790-1580是不同人的图片对
                sameflag = 1
                name = ' '.join(p[:-2])
                name1 = os.path.join(name, p[-2])
                name2 = os.path.join(name, p[-1])
            else:
                sameflag = 0
                name1, name2 = [], []  # {type}<class list>
                for word in p:
                    if type(name1) != str:
                        if '00' in word:
                            name1 = os.path.join(' '.join(name1), word)  # {type}<class 'str'>
                        else:
                            name1.append(word)
                    else:
                        if '00' in word:
                            name2 = os.path.join(' '.join(name2), word)
                        else:
                            name2.append(word)
            yield name1, name2, sameflag


def unresticted_fold_iter(dataset_path, folds_length, fold_length=1000, seed=1):
    def get_img_name(people, num):
        if num < people[1]:
            return os.path.join(people[0], 'C%05d'%(num+1))  # changed
        else:
            num -= people[1]
            return os.path.join(people[0], 'P%05d'%(num+1))

    def sample_2(num):
        a = rng.randint(num)
        b = (a + 1 + rng.randint(num-1)) % num
        return a, b
    
    rng = np.random.RandomState(seed)
    
    with open(os.path.join(
        dataset_path,
        'EvaluationProtocols',
        'FaceVerification',
        'UnRestricted',
        'UnRestrictedView2.txt',
        # 'unrestricted_p_p.txt',
        # 'unrestricted_c_c.txt',
    )) as f:
        file_lines = iter(f.readlines())

    folds_num = int(next(file_lines))
    for fold_idx in range(folds_num):
        people_num = int(next(file_lines))
        folds_length.append(fold_length*2)
        
        people_list = []
        for i in range(people_num):
            words = next(file_lines).split()
            people_list.append((' '.join(words[:-2]), int(words[-2]), int(words[-1]),))
        
        for i in range(fold_length):
            people = people_list[rng.randint(people_num)]
            a, b = sample_2(people[1] + people[2])
            yield get_img_name(people, a), get_img_name(people, b), 1

        for i in range(fold_length):
            p1, p2 = sample_2(people_num)
            p1, p2 = people_list[p1], people_list[p2]
            yield get_img_name(p1, rng.randint(p1[1]+p1[2])), get_img_name(p2, rng.randint(p2[1]+p2[2])), 0


def get_predicts(dataset_path, model_path, class_num=227, folds_iter=unresticted_fold_iter, cached=False):
    if cached:  # false
        model_name = os.path.splitext(os.path.split(model_path)[1])[0]
        feats_path = os.path.join(dataset_path, model_name+'.pkl')
        if os.path.exists(feats_path):
            predicts, folds_length = pickle.load(open(feats_path, 'rb'))
            return predicts, folds_length
    
    predicts=[]
    net = sphere20a(classnum=class_num)
    net.load_state_dict(torch.load(model_path))
    net.cuda()
    net.eval()
    net.feature = True

    transform = transforms.Compose([
        transforms.Resize((112, 96)),
    ])

    folds_length = []
    for name1, name2, sameflag in folds_iter(dataset_path, folds_length):
        img1 = os.path.join(dataset_path, 'OriginalImages', name1+'.jpg')
        # landmark1 = load_landmark(os.path.join(dataset_path, 'FacialPoints', name1+'.txt'))
        # img1 = alignment(cv2.imread(img1, 1), landmark1)
        img1 = np.array(transform(Image.open(img1).convert('RGB')))

        img2 = os.path.join(dataset_path, 'OriginalImages', name2+'.jpg')
        # landmark2 = load_landmark(os.path.join(dataset_path, 'FacialPoints', name2+'.txt'))
        # img2 = alignment(cv2.imread(img2, 1), landmark2)  # 112*96*3
        img2 = np.array(transform(Image.open(img2).convert('RGB')))

        imglist = [img1, cv2.flip(img1, 1), img2, cv2.flip(img2, 1)]  # cv2.flip(img1, 1):112*96*3
        for i in range(len(imglist)):
            imglist[i] = imglist[i].transpose(2, 0, 1).reshape((1, 3, 112, 96))
            imglist[i] = (imglist[i] - 127.5) / 128.

        img = np.vstack(imglist)  # 4*3*112*96  (垂直将imglist中的四组数据合起来)
        with torch.no_grad():
            img = Variable(torch.from_numpy(img).float(), volatile=True).cuda()
            output = net(img)  # 4*512
        f = output.data
        f1, f2 = f[0], f[2]  # img1 and img2 output
        cosdistance = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)
        predicts.append((cosdistance, sameflag))
    predicts = np.array(predicts)
    if cached:
        pickle.dump((predicts, folds_length), open(feats_path, 'wb'))
    return predicts, folds_length


def eval(predicts, folds_length, output_file=sys.stdout):
    accuracy = []
    thd = []
    far3 = []
    far2 = []
    auc = []
    folds = KFold(folds_length)
    thresholds = np.arange(-1.0, 1.0, 0.005)
    for idx, (train, test) in enumerate(folds):
        best_thresh = find_best_threshold(thresholds, predicts[train])
        accuracy.append(eval_acc(best_thresh, predicts[test]))
        thd.append(best_thresh)
        roc = calc_roc(predicts[test])
        far3.append(roc[0])
        far2.append(roc[1])
        auc.append(roc[2])
    print('WCACC={:.4f} std={:.4f} thd={:.4f}'.format(np.mean(accuracy), np.std(accuracy), np.mean(thd)), file=output_file)
    print('WCFAR3={:.4f} std={:.4f}'.format(np.mean(far3), np.std(far3)), file=output_file)
    print('WCFAR2={:.4f} std={:.4f}'.format(np.mean(far2), np.std(far2)), file=output_file)
    print('WCAUC={:.4f} std={:.4f}'.format(np.mean(auc), np.std(auc)), file=output_file, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch sphereface wc')
    parser.add_argument('--wc', default='../../datasets/WebCaricature/original_dataset', type=str)
    # parser.add_argument('--model', '-m', default='../../support_material/sphere20a.pth', type=str)
    # parser.add_argument('--model', '-m', default='outputs/init_original_dataset_stn_after_2/checkpoints/00004000.pth', type=str)
    parser.add_argument('--model', '-m', default='outputs/init_original_dataset_SE3everylr0.01/checkpoints/00004000.pth', type=str)
    # parser.add_argument('--model', '-m', default='outputs/init_original_dataset_SE3out/checkpoints/00005000.pth', type=str)
    # parser.add_argument('--model', '-m', default='outputs/init_original_dataset_SE1234every/checkpoints/00005000.pth', type=str)
    # parser.add_argument('--model', '-m', default='outputs/init_original_dataset_stn_middle/checkpoints/00005000.pth', type=str)

    parser.add_argument('--ipython', action='store_true')
    parser.add_argument('--myGpu', default='0', help='GPU Number')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.myGpu

    if args.ipython:
        from IPython import embed; embed()
        exit(0)

    predicts, folds_length = get_predicts(args.wc, args.model)

    eval(predicts, folds_length)
