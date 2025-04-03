import os
import numpy as np

from corelib import (get_name, form_node_dof, num_to_elem_loc, form_k_diag,
                     pin_jointed, form_sparse_v, sparse_cho_fac, sparse_cho_bac,
                     global_to_axial, contour_plot)


def main():
    input_file, output_file, vtk_file = get_name()
    if not os.path.exists(input_file):
        print('{}不存在'.format(input_file))
        return

    with open(input_file, 'r', encoding='utf-8') as fr, open(
            output_file, 'w', encoding='utf-8') as fw, open(
            vtk_file, 'w', encoding='utf-8') as fw2:
        num_node_on_elem = 2  # 每个单元的节点数
        num_props_comp = 2  # 材料特性数，弹性模量E、截面面积A
        penalty = 1.0e20  # 处理位移约束时使用的大数

        # 单元数、节点数、维数、材料类型数：
        num_elem, num_node, num_dim, num_prop_types = [int(x) for x in fr.readline().split()]
        num_node_dof = num_dim  # 节点自由度数，杆单元中为维数
        num_elem_dof = num_node_on_elem * num_node_dof  # 单元的自由度数

        # num_prop_types为材料类型（如钢、铝）数
        # num_props_comp为材料特性（如弹性模量、泊松比）数
        prop = np.zeros((num_prop_types, num_props_comp), dtype=np.float)
        for i in range(num_prop_types):
            line = fr.readline().split()
            for j in range(num_props_comp):
                prop[i, j] = line[j]

        # 不同单元为何种材料：
        elem_prop_type = np.ones(num_elem, dtype=np.int)
        if num_prop_types > 1:
            elem_prop_type = np.array(fr.readline().split(), dtype=np.int)

        # 节点坐标：
        g_coord = np.zeros((num_node, num_dim), dtype=np.float)
        for i in range(num_node):
            line = fr.readline().split()
            for j in range(num_dim):
                g_coord[i, j] = line[j]

        # 单元构成：
        g_num = np.zeros((num_elem, num_node_on_elem), dtype=np.int)
        for i in range(num_elem):
            line = fr.readline().split()
            for j in range(num_node_on_elem):
                g_num[i, j] = line[j]

        # 节点自由度矩阵，初始化为1，被约束的自由度置零
        node_dof = np.ones((num_node, num_node_dof), dtype=np.int)
        num_fixed_node = int(fr.readline())
        for i in range(num_fixed_node):
            line = fr.readline().split()
            for j in range(num_node_dof):
                node_dof[int(line[0])-1, j] = line[j+1]
        # 最终生成节点自由度矩阵，得到各节点自由度在整个系统中的自由度编号
        form_node_dof(node_dof)

        # 最大自由度编号即为自由度总数，也就是方程个数
        num_equation = np.max(node_dof)
        # 以一维变带宽的形式存储稀疏矩阵，刚度矩阵以向量形式存在
        # k_diag用于存储对角线元素在向量中的位置，以便计算时定位
        k_diag = np.zeros(num_equation, dtype=np.int)
        # 单元定位向量，用于获取单元中各节点的自由度是系统的第几号自由度
        elem_steer = np.zeros(num_elem_dof, dtype=np.int)
        # 整体定位向量
        global_elem_steer = np.zeros((num_elem, num_elem_dof), dtype=np.int)
        for i_elem in range(num_elem):
            num = g_num[i_elem, :]
            elem_steer = num_to_elem_loc(num, node_dof, elem_steer)
            global_elem_steer[i_elem, :] = elem_steer
            form_k_diag(k_diag, elem_steer)
        # for循环之前，k_diag存储矩阵中每一行存储几个元素
        # for循环之后，k_diag存储对角线元素在kv向量中的位置
        for i in range(1, num_equation):
            k_diag[i] += k_diag[i-1]
        fw.write('总共有{}个方程，存储带宽为{}\n'.format(num_equation, k_diag[-1]))

        # ********************************************************************* #
        # **************************** 组装刚度矩阵 ***************************** #
        # 整体刚度矩阵：
        kv = np.zeros(k_diag[-1], dtype=np.float)
        # 单元刚度矩阵：
        ke = np.zeros((num_elem_dof, num_elem_dof), dtype=np.float)
        for i_elem in range(num_elem):
            num = g_num[i_elem, :]
            coord = g_coord[num-1, :]
            ke = pin_jointed(ke, prop[elem_prop_type[i_elem]-1, 0],
                             prop[elem_prop_type[i_elem]-1, 1], coord)
            elem_steer = global_elem_steer[i_elem, :]
            form_sparse_v(kv, ke, elem_steer, k_diag)

        # ********************************************************************* #
        # ************************* 读取载荷和边界条件 *************************** #
        # 整体载荷向量，约束节点处所受载荷存储于第一位，所以数组长度比自由度多1
        loads = np.zeros(num_equation+1, dtype=np.float)
        loaded_nodes = int(fr.readline())
        for i in range(loaded_nodes):
            line = fr.readline().split()
            for j in range(num_node_dof):
                loads[node_dof[int(line[0])-1, j]] = line[j+1]
        # 受位移约束的节点数
        num_dis_node = int(fr.readline())
        if num_dis_node != 0:
            v_dis_node = np.zeros(num_dis_node, dtype=np.int)
            v_dis_dof = np.zeros(num_dis_node, dtype=np.int)
            sense_dis_node = np.zeros(num_dis_node, dtype=np.int)
            v_displacement = np.zeros(num_dis_node, dtype=np.float)
            for i in range(num_dis_node):
                v_dis_node[i], sense_dis_node[i], v_displacement[i] = fr.readline().split()
            for i in range(num_dis_node):
                v_dis_dof[i] = node_dof[v_dis_node[i]-1, sense_dis_node[i]-1]
            # 主对角线元素置大数，以处理位移约束
            kv[k_diag[v_dis_dof-1]-1] += penalty
            # 对应的载荷也同样需要置大数
            loads[v_dis_dof] = kv[k_diag[v_dis_dof-1]-1] * v_displacement

        # ********************************************************************* #
        # ****************************** 方程求解 ****************************** #
        sparse_cho_fac(kv, k_diag)
        sparse_cho_bac(kv, loads, k_diag)
        loads[0] = 0

        # ********************************************************************* #
        # ****************************** 结果处理 ****************************** #
        fw2.write('# vtk DataFile Version 3.0\n')
        fw2.write('Bar Element\n')
        fw2.write('ASCII\n')

        fw2.write('\nDATASET UNSTRUCTURED_GRID\n')

        fw2.write('\nPOINTS {} float\n'.format(num_node))
        for i in range(num_node):
            if num_dim == 1:
                fw2.write('{} 0.0 0.0\n'.format(g_coord[i, 0]))
            elif num_dim == 2:
                fw2.write('{} {} 0.0\n'.format(g_coord[i, 0], g_coord[i, 1]))
            elif num_dim == 3:
                fw2.write('{} {} {}\n'.format(g_coord[i, 0], g_coord[i, 1], g_coord[i, 2]))

        fw2.write('\nCELLS {} {}\n'.format(num_elem, 3*num_elem))
        for i_elem in range(num_elem):
            fw2.write('2 {} {}\n'.format(g_num[i_elem, 0]-1, g_num[i_elem, 1]-1))

        fw2.write('\nCELL_TYPES {}\n'.format(num_elem))
        for i_elem in range(num_elem):
            fw2.write('3\n')

        fw.write('节点      位移\n')
        fw2.write('\nPOINT_DATA {}\n'.format(num_node))
        fw2.write('VECTORS Displacement float\n')
        for i in range(num_node):
            fw.write('{:4} '.format(i+1))
            for j in range(num_node_dof):
                fw.write('{:12.4e} '.format(loads[node_dof[i, j]]))
                fw2.write('{:12.4e} '.format(loads[node_dof[i, j]]))
            fw.write('\n')
            if num_node_dof == 1:
                fw2.write('0.0 0.0\n')
            elif num_node_dof == 2:
                fw2.write('0.0\n')
            elif num_node_dof == 3:
                fw2.write('\n')

        fw.write('单元    载荷\n')
        for i_elem in range(num_elem):
            num = g_num[i_elem, :]
            coord = g_coord[num-1, :]
            ke = pin_jointed(ke, prop[elem_prop_type[i_elem]-1, 0],
                             prop[elem_prop_type[i_elem]-1, 1], coord)
            elem_steer = global_elem_steer[i_elem, :]
            elem_dis = loads[elem_steer]
            action = np.dot(ke, elem_dis)
            fw.write('{:4}'.format(i_elem+1))
            for i in action:
                fw.write('{:12.4e} '.format(i))
            fw.write('\n')
            axial = global_to_axial(action, coord)
            fw.write('      轴向力：{:12.4e}\n'.format(axial))

    # 显示结果
    contour_plot(vtk_file, 'Displacement', -1, 100)
if __name__ == '__main__':
 main()