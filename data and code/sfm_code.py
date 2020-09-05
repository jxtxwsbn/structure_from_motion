import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt
import open3d as o3d
import math
from numpy import linalg as LA
from scipy.sparse import lil_matrix
import time
from scipy.optimize import least_squares

#global paramters
cameraIntrinsic = np.array([[1520.400000, 0.000000, 302.320000], [0.000000, 1525.900000, 246.870000], [0.000000, 0.000000, 1.000000]])
cameraIntrinsic_vector = np.array([1520.4,1525.9,302.32,246.87],dtype=float)
R0 = np.eye(3, 3)
T0 = np.zeros((3, 1))
# functions
def keypoint_match(kp1, kp2, des1, des2, reject_outliers=False, threshold = 10):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    crct_matches = list()
    # ratio test as per Lowe's paper
    for i in range(0, len(matches)):
        if len(matches[i]) != 2:
            continue
        m, n = matches[i]
        if m.distance < 0.85 * n.distance:
            crct_matches.append(m)

    crct_matches_no_outliers = list()
    if reject_outliers:
        for i, match in enumerate(crct_matches):
            prev = kp1[match.queryIdx].pt
            curr = kp2[match.trainIdx].pt
            dist = math.sqrt((prev[0]-curr[0])**2 + (prev[1] - curr[1])**2)
            if dist <= threshold:
                crct_matches_no_outliers.append(match)
        return  crct_matches_no_outliers
    else:
        return  crct_matches

def estimate_essential_matrix(matches, kp_prev, kp_curr):
    points1 = []
    points2 = []
    for index in range(0, len(matches)):
        points1.append(kp_prev[matches[index].queryIdx].pt)
        points2.append(kp_curr[matches[index].trainIdx].pt)
    points1 = np.int32(points1)
    points2 = np.int32(points2)
    E, mask = cv2.findEssentialMat(points1, points2, cameraIntrinsic, cv2.RANSAC, prob = 0.9, 	threshold = 5.0)
    #Select inlier points
    points1 = points1[mask.ravel()==1]
    points2 = points2[mask.ravel()==1]
    return E, mask, points1, points2 #mask needed in the future

#get the pixel coordinate for matched points
def get_match_point_uv(kp1,kp2,matches):
    points1= np.asarray([kp1[m.queryIdx].pt for m in matches])
    points2 = np.asarray([kp2[m.trainIdx].pt for m in matches])
    return points1, points2

def detCameraPose(EssentialMat, points1, points2):
    retval, R, T, mask = cv2.recoverPose(EssentialMat, points1, points2,cameraIntrinsic)
    return R, T, mask #mask need for removing some outliers wrt [R|T]

def maskout(points1,points2,mask):
    points1 = points1[mask.ravel() > 0]
    points2 = points2[mask.ravel() > 0]
    return points1, points2

#linear triangulaton 2 sets of pixel coordinates into aone set of 3d points
def linear_triangulation(K, R1, T1, R2, T2, uv1, uv2):
    RT1 = np.concatenate((R1, T1), axis=1)
    RT2 = np.concatenate((R2, T2), axis=1)
    proj1 = np.dot(K, RT1)
    proj2 = np.dot(K, RT2)
    proj1 = np.float32(proj1)
    proj2 = np.float32(proj2)
    uv1 = np.float32(uv1.T)
    uv2 = np.float32(uv2.T)
    s = cv2.triangulatePoints(proj1, proj2, uv1, uv2)
    X = s/s[3]
    uv1_recover=np.dot(proj1[:3],X)
    uv1_recover /= uv1_recover[2]
    point = cv2.convertPointsFromHomogeneous(s.T)
    return np.array(point)

# initial 3d-point structure from the first two images
def initial_structure(cameraIntrinsic, key_point_list, match_list, points1, points2):
    R0 = np.eye(3, 3)
    T0 = np.zeros((3, 1))
    structrue = linear_triangulation(cameraIntrinsic, R0, T0, R, T, points1, points2)
    rotations = [R0, R]
    translations = [T0, T]
    relative_list = []  # key points corresponding to 3d points (if the index for one key points is 20 in the relative-array,the 20th 3d points comes from the kep point and its match
    for key_points in key_point_list:
        relative_list.append(np.zeros((len(key_points))) - 1)
    relative_list = np.array(relative_list)
    idx = 0
    matches = match_list[0]
    j=-1
    for i, match in enumerate(matches):
        if mask1[i] == 0:
            continue
        if mask1[i] != 0:
            j=j+1
            if mask2[j] ==0:
                continue
        relative_list[0][match.queryIdx] = idx
        relative_list[1][match.trainIdx] = idx
        idx += 1
    return structrue, relative_list, rotations, translations

# get the PnP inputs from existing structure of the 3-d points and their corresponding uv in the new image
def get_pnp_input(matches, relative1, structure, key_points):
    d2_points = np.array([])
    d3_points = np.array([])
    for match in matches:
        structure_idx = relative1[match.queryIdx]
        if structure_idx < 0:
            continue
        d2_points = np.append(d2_points,key_points[match.trainIdx].pt)
        d3_points = np.append(d3_points,structure[int(structure_idx)])
    return d2_points.reshape(-1,2), d3_points.reshape(-1,3)

#linear triangulation for new pair of images to get a set of new 3d points
def new_points_from_pnp(d2_points, d3_points, cameraIntrinsic, kp1, kp2, matches):
    c, r, t, _ = cv2.solvePnPRansac(d3_points, d2_points, cameraIntrinsic, distCoeffs=None, iterationsCount=2000,
                                    reprojectionError=6, confidence=0.999, useExtrinsicGuess=False)
    #cv2.solvePnP()
    R, _ = cv2.Rodrigues(r)
    rotations.append(R)
    translations.append(t)
    uv1, uv2 = get_match_point_uv(kp1, kp2, matches)
    new_points = linear_triangulation(cameraIntrinsic, rotations[i], translations[i], rotations[i+1], translations[i+1], uv1, uv2)###
    return new_points

# fuse the new 3d point into the existing structure
def update_structure(structure, new_points, relative1, relative2, matches):
    for i, match in enumerate(matches):
        if relative1[match.queryIdx] >= 0:
            relative2[match.trainIdx] = relative1[match.queryIdx]
            continue
        structure = np.append(structure, [new_points[i]], axis=0)
        relative1[match.queryIdx] = len(structure) - 1
        relative2[match.trainIdx] = len(structure) - 1
    return structure, relative1, relative2

def visualize(pointCloud):
    #input: pointCloud nparray with dimensions nx3
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointCloud)
    o3d.io.write_point_cloud("./sync.ply", pcd)

    # Load saved point cloud and visualize it
    pcd_load = o3d.io.read_point_cloud("./sync.ply")
    o3d.visualization.draw_geometries([pcd_load])


def error_calcalate(d3_p,d2_p,r,t,cameraIntrinsic):
    p, J = cv2.projectPoints(d3_p.reshape(1, 1, 3), r, t, cameraIntrinsic, np.array([]))
    # print(p)
    p = p.reshape(2)
    e = np.sum(np.power((d2_p - p),2))/2
    # print(e)
    return e

def sort_category(rotations, motions, cameraIntrinsic, relative_list, kp_list, structure):
    '''sort the 3d points into three category based on reprojection error'''
    r_vctor_list=[]
    category_list =[]
    for i in range(len(rotations)):
        r, _ = cv2.Rodrigues(rotations[i])
        r_vctor_list.append(r)
    for key_points in kp_list:
        category_list.append(np.zeros((len(key_points))) - 1)
    category_list = np.array(category_list)
    for i in range(len(rotations)):
        point3d_ids = relative_list[i]
        key_points = kp_list[i]
        r = r_vctor_list[i]
        t = motions[i]
        category = category_list[i]
        for j in range(len(key_points)):
            point3d_id = int(point3d_ids[j])
            if point3d_id < 0:
                continue
            error = error_calcalate(structure[point3d_id], key_points[j].pt, r, t, cameraIntrinsic)
            if error <=5:#no need for BA. category 0
                category[j]=0
            if 5<error<300:#for bundle adjustment. category 1
                category[j]=1
    return category_list

def get_fine_points(category_list,relative_list,structure):
    '''get 3d-points whose reprojection error is less than 5'''
    fine_points = np.array([])
    fine_idx = np.array([])
    for i in range(len(category_list)):
        category =category_list[i]
        relative = relative_list[i]
        for j in range(len(category)):
            if category[j]==0 and int(relative[j]) not in fine_idx:
                fine_points = np.append(fine_points,structure[int(relative[j])])
                fine_idx = np.append(fine_idx,int(relative[j]))
    return fine_points.reshape(-1,3),fine_idx

#below is the functions used for bundle adjustments
def get_parameter(relative_list,kp_list,category_list):
    points_2d = np.array([])
    points_ind = np.array([])
    camera_ind = np.array([])
    for i in range(0,len(images)):
        rel = relative_list[i].astype(int)
        kep = kp_list[i]
        category = category_list[i]
        for j in range(rel.shape[0]):
            if rel[j]<0:
                continue
            #if category[j]==1:#or category[j]==0
            points_ind = np.append(points_ind,rel[j])
            camera_ind = np.append(camera_ind,np.array([i]))
            points_2d = np.append(points_2d,kep[j].pt)
    return points_ind.astype(int),camera_ind.astype(int),points_2d.reshape(-1,2)

def get_camera_params(rotations, translations):
    camera_params = np.array([])
    for i in range(len(rotations)):
        r_vec, _ = cv2.Rodrigues(rotations[i])
        r_vec = r_vec.reshape(-1,)
        trans_vec = translations[i].reshape(-1,)
        row = np.append(r_vec,trans_vec)
        camera_params = np.append(camera_params,row)
        camera_params = np.expand_dims(camera_params,axis=0)
    return camera_params.reshape(len(rotations),-1)

def rotate(points,rot_vecs):
    """Rotate points by given rotation vectors.
    Rodrigues' rotation formula is used."""
    theta = np.linalg.norm(rot_vecs,axis=1)[:,np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs/theta
        v = np.nan_to_num(v)
    dot = np.sum(points*v,axis=1)[:,np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v

def project(points,camera_params,camera_vector):
    """Convert 3-D points to 2-D by projecting onto images."""
    points_proj = rotate(points,camera_params[:,:3])
    points_proj += camera_params[:,3:6]
    points_proj = points_proj[:,:2]/points_proj[:,2,np.newaxis]
    f = camera_vector[:2][np.newaxis,:]
    c = camera_vector[2:][np.newaxis,:]
    points_proj = points_proj*f + c
    return points_proj

def fun(params, n_cameras, n_points, camera_ind, points_ind, points_2d,camera_vector):
    """Compute residuals."""
    camera_params = params[:n_cameras*6].reshape((n_cameras,6))
    points_3d = params[n_cameras*6:].reshape(n_points,3)
    points_pro = project(points_3d[points_ind],camera_params[camera_ind],camera_vector)
    return(points_pro - points_2d).ravel()

def bundle_adjustments(n_cameras, n_points, camera_indices, points_indices):
    m = camera_indices.size*2
    n = n_cameras*6 +n_points*3
    A = lil_matrix((m,n),dtype=int)
    i = np.arange(camera_indices.size)
    for s in range(6):
        A[2*i,camera_indices*6+s]=1
        A[2*i+1,camera_indices*6+s]=1
    for s in range(3):
        A[2*i,n_cameras*6+points_indices*3+s]=1
        A[2*i+1,n_cameras*6+points_indices*3+s]=1
    return A

def more_accuracy(points_ind,fun_results,points_3d):
    results = np.array([])
    for i in range(len(points_ind)):
        error = np.sum(np.power(fun_results[2*i:2*(i+1)],2))/2
        if error > 30:
            points_3d[points_ind[i]]=0
    for i in range(points_3d.shape[0]):
        if np.linalg.norm(points_3d[i])!=0:
            results = np.append(results,points_3d[i])
    return results.reshape(-1,3)

if __name__ == "__main__":
    #data = input('Please enter 1 for sparse temple data and 2 for no sparse temple data:\n')
    #if data == '1':
    path = './templeSparseRing/*png'
    #if data == '2':
        #path = 'temple/*png'
    images = [cv2.imread(file, 0) for file in glob.glob(path)]  # create a image dir
    kp_list = []  # kep_points list
    des_list = []  # descriptor list
    match_list = []  # match list
    sift = cv2.xfeatures2d.SIFT_create()
    for i in range(len(images)):
        img = images[i]
        kp_curr, des_curr = sift.detectAndCompute(img, None)
        kp_list.append(kp_curr)  # totally 16 sets of kp
        des_list.append(des_curr)
    for i in range(len(images) - 1):
        des_prev = des_list[i]
        des_curr = des_list[i + 1]
        matches = keypoint_match(kp_list[i], kp_list[i + 1], des_list[i], des_list[i + 1], reject_outliers=False, )
        match_list.append(matches)

    match_list = np.array(match_list)
    kp_list = np.array(kp_list)
    des_list = np.array(des_list)
    EssentialMat, mask1, points1, points2 = estimate_essential_matrix(match_list[0], kp_list[0], kp_list[1])
    print('p1,p2 from essential', len(points1), len(points2))
    R, T, mask2 = detCameraPose(EssentialMat, points1, points2)
    points1, points2 = maskout(points1, points2, mask2)  # remove outlier points
    print('p1,p2 from pose mask', len(points1), len(points2))
    # get the initial structure
    structure, relative_list, rotations, translations = initial_structure(cameraIntrinsic, kp_list, match_list, points1,
                                                                          points2)
    # points = linear_triangulation(cameraIntrinsic, R0, T0, R, T, points1, points2)
    # points = points.squeeze()
    print('initial_structure', structure.shape)

    print('the struture from the first two images')
    visualize(structure.squeeze())
    # pnp for the remaining images
    number = input(" input 'ENTER ' or the images number() for sfm, the min is 3, the max is 16 and the default is 3:\n ")
    if number == '':
        number = 3
    number = int(number)-1
    for i in range(1, number):
        d2_points, d3_points = get_pnp_input(match_list[i], relative_list[i], structure, kp_list[i + 1])
        print('number of points for solve PNP', len(d2_points))
        if len(d2_points) < 7:  # solvePnp need at least 7 points
            while len(d2_points) < 7:
                d2_points = np.append(d2_points, [d2_points[0]], axis=0)
                d3_points = np.append(d3_points, [d3_points[0]], axis=0)
                print('not enough points to solve pnp')
        new_points = new_points_from_pnp(d2_points, d3_points, cameraIntrinsic, kp_list[i], kp_list[i + 1],
                                         match_list[i])
        # update the structure
        structure, relative_list[i], relative_list[i + 1] = update_structure(structure, new_points, relative_list[i],
                                                                             relative_list[i + 1], match_list[i], )
        print('structure befor bundle adjustment')
        structure = structure.squeeze()
        print(structure.shape)
        #visualize(structure)
        # refine the structure
        category_list = sort_category(rotations, translations, cameraIntrinsic, relative_list, kp_list, structure)
        fine_points, fine_idx = get_fine_points(category_list, relative_list, structure) # fine_points has a smaller reprojection error before BA
        print('fine_points')
        #visualize(fine_points)
        points_ind, camera_ind, points_2d = get_parameter(relative_list, kp_list, category_list)
        points_3d = structure
        camera_params = get_camera_params(rotations, translations)
        n_cameras = camera_params.shape[0]
        n_points = points_3d.shape[0]
        x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
        f0 = fun(x0, n_cameras, n_points, camera_ind, points_ind, points_2d, cameraIntrinsic_vector)
        if i == 1:
            plt.plot(f0)
            plt.title('reprojection error befor BA')
            plt.show()
        A = bundle_adjustments(n_cameras, n_points, camera_ind, points_ind)
        t0 = time.time()
        res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-6, method='trf', loss='cauchy',
                            max_nfev=1000,
                            args=(n_cameras, n_points, camera_ind, points_ind, points_2d, cameraIntrinsic_vector))
        t1 = time.time()
        print('optimization took {0:.0f} seconds'.format(t1 - t0))
        if i == number - 1:
            plt.plot(res.fun)
            plt.title('reprojection error after BA')
            plt.show()
        points_final = res.x[len(camera_params.ravel()):].reshape(-1, 3)
        #visualize(points_final)
        cameras_mat = res.x[:len(camera_params.ravel())].reshape(n_cameras, -1)
        for j in range(cameras_mat.shape[0]):
            r_v1 = camera_params[j, 0:3]
            R_m1, _ = cv2.Rodrigues(r_v1)
            t1 = camera_params[j, 3:6][:, np.newaxis]
            if j == cameras_mat.shape[0]:
                rotations[i]=R_m1
                translations[i] = t1
        structure = points_final
        structure[fine_idx.astype(int)]= fine_points
        structure = np.expand_dims(structure,axis=1)
    visualize(structure.squeeze())#suceess
    visualize(points_final)

    inliers = more_accuracy(points_ind,res.fun,points_final)
    inliers = np.concatenate((inliers,fine_points),axis=0)
    #visualize(inliers)