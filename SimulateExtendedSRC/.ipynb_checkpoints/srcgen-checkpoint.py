#!/home/innov2021/miniconda3/envs/3ml/bin/python3.11
import numpy as np
import matplotlib.pyplot as plt
import scipy

def randomsrc(maxstep, xsize):
    np.random.seed()
    img = np.zeros((xsize, xsize))
    cp = int(xsize/2)
    x = [(np.random.randint(cp-4,cp+4), np.random.randint(cp-4,cp+4))]
    vx = 1 
    vy = 1
    dt = 1
    steps = np.random.randint(0,maxstep)
    theta = np.random.uniform(0,2 * np.pi)
    for i in range(steps):
        a = np.random.uniform(0.1,2)
        dtheta = np.random.uniform(-np.pi/2, np.pi/2)
        theta += dtheta
        vx += a * np.cos(theta) * dt
        vy += a * np.sin(theta) * dt
        v = (vx**2 + vy**2)**0.5 * np.random.uniform(0.5,3)
        vx /= v
        vy /= v
        x.append((x[-1][0] + vx * dt, x[-1][1] + vy * dt ))
    
    # img = np.zeros((xsize, xsize))
    d2f = 0
    df = 0
    f = 1
    for i,x in enumerate(x):
        if (x[0] < 0) | (x[0] > 63):
            continue
        if (x[1] < 0) | (x[1] > 63):
            continue
        d2f = np.random.uniform(-10,10)
        f += d2f
        f = np.abs(f)
        img[int(x[0]), int(x[1])] += float(f)
    weights = np.random.uniform(0,1,(3,3))
    img = scipy.ndimage.gaussian_filter(img,sigma=np.random.uniform(0.1,steps / maxstep *2))
    return img

def srcgen(nsrc, xsize):
    img = None
    for i in range(np.random.randint(nsrc)+1):
        maxstep = np.random.uniform(1,100)
        if img is None:
            img = randomsrc(maxstep, xsize)
        else:
            img += randomsrc(maxstep, xsize)
    return img


if __name__=="__main__":
    img = srcgen(5,100)
    plt.imshow(img)