import os, sys
import numpy as np
import cv2



# Based on https://github.com/youngLBW/HRN/blob/main/util/util_.py#L398
def read_obj(obj_path, print_shape=False):
    with open(obj_path, 'r') as f:
        bfm_lines = f.readlines()

    vertices = []
    faces = []
    uvs = []
    vns = []
    faces_uv = []
    faces_normal = []
    max_face_length = 0
    for line in bfm_lines:
        if line[:2] == 'v ':
            vertex = [float(a) for a in line.strip().split(' ')[1:] if len(a)>0]
            vertices.append(vertex)

        if line[:2] == 'f ':
            items = line.strip().split(' ')[1:]
            face = [int(a.split('/')[0]) for a in items if len(a)>0]
            max_face_length = max(max_face_length, len(face))
            # if len(faces) > 0 and len(face) != len(faces[0]):
            #     continue
            faces.append(face)

            if '/' in items[0] and len(items[0].split('/')[1])>0:
                face_uv = [int(a.split('/')[1]) for a in items if len(a)>0]
                faces_uv.append(face_uv)

            if '/' in items[0] and len(items[0].split('/')) >= 3 and len(items[0].split('/')[2])>0:
                face_normal = [int(a.split('/')[2]) for a in items if len(a)>0]
                faces_normal.append(face_normal)

        if line[:3] == 'vt ':
            items = line.strip().split(' ')[1:]
            uv = [float(a) for a in items if len(a)>0]
            uvs.append(uv)

        if line[:3] == 'vn ':
            items = line.strip().split(' ')[1:]
            vn = [float(a) for a in items if len(a)>0]
            vns.append(vn)

    vertices = np.array(vertices).astype(np.float32)
    if max_face_length <= 3:
        faces = np.array(faces).astype(np.int32)
    else:
        print('not a triangle face mesh!')

    if vertices.shape[1] == 3:
        mesh = {
            'vertices': vertices,
            'faces': faces,
        }
    else:
        mesh = {
            'vertices': vertices[:, :3],
            'colors': vertices[:, 3:],
            'faces': faces,
        }

    if len(uvs) > 0:
        uvs = np.array(uvs).astype(np.float32)
        mesh['UVs'] = uvs

    if len(vns) > 0:
        vns = np.array(vns).astype(np.float32)
        mesh['normals'] = vns

    if len(faces_uv) > 0:
        if max_face_length <= 3:
            faces_uv = np.array(faces_uv).astype(np.int32)
        mesh['faces_uv'] = faces_uv

    if len(faces_normal) > 0:
        if max_face_length <= 3:
            faces_normal = np.array(faces_normal).astype(np.int32)
        mesh['faces_normal'] = faces_normal

    if print_shape:
        print('num of vertices', len(vertices))
        print('num of faces', len(faces))
    return mesh


# Based on https://github.com/youngLBW/HRN/blob/main/util/util_.py#L343
def write_obj(save_path, vertices, faces=None, UVs=None, faces_uv=None, normals=None, faces_normal=None, texture_map=None, save_mtl=False, vertices_color=None):
    save_dir = os.path.dirname(save_path)
    save_name = os.path.splitext(os.path.basename(save_path))[0]

    if save_mtl or texture_map is not None:
        if texture_map is not None:
            cv2.imwrite(os.path.join(save_dir, save_name + '.jpg'), texture_map)
        with open(os.path.join(save_dir, save_name + '.mtl'), 'w') as wf:
            wf.write('# Created by HRN\n')
            wf.write('newmtl material_0\n')
            wf.write('Ka 1.000000 0.000000 0.000000\n')
            wf.write('Kd 1.000000 1.000000 1.000000\n')
            wf.write('Ks 0.000000 0.000000 0.000000\n')
            wf.write('Tr 0.000000\n')
            wf.write('illum 0\n')
            wf.write('Ns 0.000000\n')
            wf.write('map_Kd {}\n'.format(save_name + '.jpg'))

    with open(save_path, 'w') as wf:
        if save_mtl or texture_map is not None:
            wf.write("# Create by HRN\n")
            wf.write("mtllib ./{}.mtl\n".format(save_name))

        if vertices_color is not None:
            for i, v in enumerate(vertices):
                wf.write('v {} {} {} {} {} {}\n'.format(v[0], v[1], v[2], vertices_color[i][0], vertices_color[i][1], vertices_color[i][2]))
        else:
            for v in vertices:
                wf.write('v {} {} {}\n'.format(v[0], v[1], v[2]))

        if UVs is not None:
            for uv in UVs:
                wf.write('vt {} {}\n'.format(uv[0], uv[1]))

        if normals is not None:
            for vn in normals:
                wf.write('vn {} {} {}\n'.format(vn[0], vn[1], vn[2]))

        if faces is not None:
            for ind, face in enumerate(faces):
                if faces_uv is not None or faces_normal is not None:
                    if faces_uv is not None:
                        face_uv = faces_uv[ind]
                    else:
                        face_uv = face
                    if faces_normal is not None:
                        face_normal = faces_normal[ind]
                    else:
                        face_normal = face
                    row = 'f ' + ' '.join(['{}/{}/{}'.format(face[i], face_uv[i], face_normal[i]) for i in range(len(face))]) + '\n'
                else:
                    row = 'f ' + ' '.join(['{}'.format(face[i]) for i in range(len(face))]) + '\n'
                wf.write(row)