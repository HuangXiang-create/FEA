import numpy as np
import vtk

def contour_plot(vtk_filename, vector_name, component=-1, factor=1):
    """
    绘制云图
    :param vtk_filename: vtk格式的结果文件名
    :param vector_name: 需要显示的场变量
    :param component: 场的分量，-1为幅值，0、1、2分别为向量的三个分量
    :param factor: 变形缩放系数
    """
    if vtk_filename.endswith('.vtu'):
        reader = vtk.vtkUnstructuredGridReader()
    else:
        print('不能识别的文件格式。')
        return

    line_width = 3
    if component < -1 or component > 2:
        component = -1

    reader.SetFileName(vtk_filename)
    # 默认显示物体变形后的位形。
    # 读取所有的标量场和向量场，以便能够在变形图上显示其它云图
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    # 激活位移向量，以便通过warpVector计算出物体变形后的形态
    reader.SetVectorsName('Displacement')
    reader.Update()

    ren = vtk.vtkRenderer()
    ren_win = vtk.vtkRenderWindow()
    ren_win.AddRenderer(ren)
    ren_win.SetSize(900, 720)
    ren_win.SetWindowName('PyPost')
    ren_win.Render()
    iren = vtk.vtkRenderWindowInteractor()
    style = vtk.vtkInteractorStyleTrackballCamera()
    iren.SetRenderWindow(ren_win)
    iren.SetInteractorStyle(style)

    lut = vtk.vtkLookupTable()
    lut.SetHueRange(0.6667, 0.0)  # 让场变量小的地方显示绿色，大的显示红色。默认情况与之相反
    if component == -1:
        lut.SetVectorModeToMagnitude()
    else:
        lut.SetVectorModeToComponent()
        lut.SetVectorComponent(component)

    # 坐标轴
    axes = vtk.vtkAxesActor()
    axes_widget = vtk.vtkOrientationMarkerWidget()
    axes_widget.SetOrientationMarker(axes)
    axes_widget.SetInteractor(iren)
    axes_widget.SetViewport(0.0, 0.0, 0.15, 0.2)
    axes_widget.SetEnabled(1)
    axes_widget.InteractiveOff()

    # 原始形状
    original_mapper = vtk.vtkDataSetMapper()
    original_mapper.SetInputData(reader.GetOutput())
    original_actor = vtk.vtkActor()
    original_actor.SetMapper(original_mapper)
    original_actor.GetProperty().SetLineWidth(line_width)
    original_actor.GetProperty().SetColor(0.5, 0.5, 0.5)
    original_actor.GetProperty().EdgeVisibilityOn()
    ren.AddActor(original_actor)

    # # 原始形状轮廓
    # outline = vtk.vtkOutlineFilter()
    # outline.SetInputConnection(reader.GetOutputPort())
    # outline_mapper = vtk.vtkDataSetMapper()
    # outline_mapper.SetInputConnection(outline.GetOutputPort())
    # outline_actor = vtk.vtkActor()
    # outline_actor.SetMapper(outline_mapper)
    # outline_actor.GetProperty().SetLineWidth(line_width)
    # outline_actor.GetProperty().SetColor(0.5, 0.5, 0.5)
    # ren.AddActor(outline_actor)

    # 变形后的形状
    warp = vtk.vtkWarpVector()
    warp.SetInputConnection(reader.GetOutputPort())
    deformation_mapper = vtk.vtkDataSetMapper()
    deformation_mapper.SetInputConnection(warp.GetOutputPort())
    deformation_mapper.SetLookupTable(lut)
    if vector_name in ['Displacement']:
        deformation_mapper.SetScalarModeToUsePointFieldData()
        vectors = reader.GetOutput().GetPointData().GetArray(vector_name)
    else:
        deformation_mapper.SetScalarModeToUseCellFieldData()
        vectors = reader.GetOutput().GetCellData().GetArray(vector_name)
    # 在变形后的形态上显示选择的场变量
    deformation_mapper.SelectColorArray(vector_name)
    deformation_mapper.SetScalarRange(vectors.GetRange(component))
    deformation_actor = vtk.vtkActor()
    deformation_actor.SetMapper(deformation_mapper)
    deformation_actor.GetProperty().SetLineWidth(line_width)
    warp.SetScaleFactor(factor)
    ren.AddActor(deformation_actor)

    # 图例
    scalar_bar = vtk.vtkScalarBarActor()
    scalar_bar.SetLookupTable(deformation_mapper.GetLookupTable())
    scalar_bar.SetTitle('{}\n'.format(vector_name))
    scalar_bar.SetNumberOfLabels(8)
    scalar_bar.SetPosition(0.05, 0.2)
    scalar_bar.SetPosition2(0.1, 0.75)
    scalar_bar.SetLabelFormat("%5.3e")
    prop_title = vtk.vtkTextProperty()
    prop_label = vtk.vtkTextProperty()
    prop_title.SetFontFamilyToArial()
    prop_title.ItalicOff()
    prop_title.BoldOn()
    prop_title.SetColor(0.1, 0.1, 0.1)
    prop_label.BoldOff()
    prop_label.SetColor(0.1, 0.1, 0.1)
    scalar_bar.SetTitleTextProperty(prop_title)
    scalar_bar.SetLabelTextProperty(prop_label)
    ren.AddActor(scalar_bar)

    ren.SetBackground(0.9, 0.9, 0.9)
    ren.ResetCamera()

    iren.Initialize()
    iren.Start()


def form_k_diag(k_diag, elem_steer):
    """
    :param k_diag: 对角线辅助向量
    :param elem_steer: 单元定位向量
    :return: 刚度矩阵中每一行需要存储的元素个数

    一维变带宽存储时，向量kv只存储刚度矩阵中的部分数据，即：
    从每一行的第一个非零元素开始，到对角线元素结束。
    令k=elem_steer[i]，则k为系统的第k个自由度，也就是矩阵的第k行，
    而elem_steer中的其它自由度，则是矩阵中同一行的其它元素。
    一行中最大自由度编号减最小自由度编号再加一，即为该行需要存储的元素个数。
    """
    i_dof = np.size(elem_steer)

    # 遍历单元，自由度相减并加一。
    for i in range(i_dof):
        iwp1 = 1
        if elem_steer[i] != 0:
            for j in range(i_dof):
                if elem_steer[j] != 0:
                    im = elem_steer[i] - elem_steer[j] + 1
                    if im > iwp1:
                        iwp1 = im
            # 由于索引从0开始，所以对k_diag向量的索引需要减一
            k = elem_steer[i]
            if iwp1 > k_diag[k-1]:
                k_diag[k-1] = iwp1


def form_node_dof(node_dof):
    """
    生成节点自由度矩阵，存储节点自由度为整体的第几个自由度
    :param node_dof: 节点自由度矩阵，被约束的自由度置零
    """
    m = 0
    for i in range(np.size(node_dof, 0)):
        for j in range(np.size(node_dof, 1)):
            if node_dof[i][j] != 0:
                m += 1
                node_dof[i][j] = m


def form_sparse_v(kv, ke, elem_steer, k_diag):
    """
    组装整体刚度矩阵，以一维变带宽方式存储。
    :param kv: 向量形式的整体刚度矩阵
    :param ke: 单元刚度矩阵
    :param elem_steer: 单元定位向量
    :param k_diag: 对角线辅助向量

    刚度矩阵k和向量kv的关系为：k[i,j] = kv[k_diag[i]-i+j]
    由于索引从0开始，所以索引时需要减一
    """
    # elem_steer[i]即为上面公式中的i
    # elem_steer[j]即为上面公式中的j
    i_dof = np.size(elem_steer, 0)
    for i in range(i_dof):
        if elem_steer[i] != 0:
            for j in range(i_dof):
                if elem_steer[j] != 0:
                    if elem_steer[i] - elem_steer[j] >= 0:
                        i_val = k_diag[elem_steer[i]-1] - elem_steer[i] + elem_steer[j]
                        kv[i_val-1] += ke[i, j]


def get_name():
    """
    :return: 输入文件名和输出文件名
    """
    import sys
    if len(sys.argv) < 2:
        filename = input('请输入文件名：\n')
    else:
        filename = sys.argv[1]
    if filename.split('.')[-1] == 'inp':
        filename = '.'.join(filename.split('.')[:-1])
    input_file = '{}.inp'.format(filename)
    output_file = '{}.out'.format(filename)
    vtk_file = '{}.vtu'.format(filename)
    return input_file, output_file, vtk_file


def global_to_axial(action, coord):
    num_dim = np.size(coord, 1)
    add = 0.0
    for i in range(num_dim):
        add += (coord[1, i] - coord[0, i]) ** 2
    length = np.sqrt(add)
    axial = 0.0
    for i in range(num_dim):
        axial += (coord[1, i] - coord[0, i]) / length * action[num_dim+i]
    return axial


def num_to_elem_loc(num, node_dof, elem_steer):
    """
    :param num: 单元节点编号
    :param node_dof: 节点自由度矩阵
    :param elem_steer: 初始化后的单元定位向量
    :return: elem_steer, 实际的单元定位向量
    """
    num_node_in_elem = np.size(num, 0)
    num_node_dof = np.size(node_dof, 1)
    for i in range(num_node_in_elem):
        k = i * num_node_dof
        elem_steer[k: k+num_node_dof] = node_dof[num[i]-1, :]
    return elem_steer


def pin_jointed(ke, e, a, coord):
    num_dim = np.size(coord, 1)
    if num_dim == 1:
        length = coord[1, 0] - coord[0, 0]
        ke[0, 0] = 1.0
        ke[0, 1] = -1.0
        ke[1, 0] = -1.0
        ke[1, 1] = 1.0
    elif num_dim == 2:
        x1 = coord[0, 0]
        y1 = coord[0, 1]
        x2 = coord[1, 0]
        y2 = coord[1, 1]
        length = np.sqrt((y2-y1)**2 + (x2-x1)**2)
        cs = (x2-x1)/length
        sn = (y2-y1)/length
        ll = cs * cs
        mm = sn * sn
        lm = cs * sn
        ke[0, 0] = ll
        ke[2, 2] = ll
        ke[0, 2] = -ll
        ke[2, 0] = -ll
        ke[1, 1] = mm
        ke[3, 3] = mm
        ke[1, 3] = -mm
        ke[3, 1] = -mm
        ke[0, 1] = lm
        ke[1, 0] = lm
        ke[2, 3] = lm
        ke[3, 2] = lm
        ke[0, 3] = -lm
        ke[3, 0] = -lm
        ke[1, 2] = -lm
        ke[2, 1] = -lm
    elif num_dim == 3:
        x1 = coord[0, 0]
        y1 = coord[0, 1]
        z1 = coord[0, 2]
        x2 = coord[1, 0]
        y2 = coord[1, 1]
        z2 = coord[1, 2]
        length = np.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
        l = (x2 - x1) / length
        m = (y2 - y1) / length
        n = (z2 - z1) / length
        ll = l * l
        mm = m * m
        nn = n * n
        lm = l * m
        mn = m * n
        ln = l * n
        ke[0, 0] = ll
        ke[3, 3] = ll
        ke[1, 1] = mm
        ke[4, 4] = mm
        ke[2, 2] = nn
        ke[5, 5] = nn
        ke[0, 1] = lm
        ke[1, 0] = lm
        ke[3, 4] = lm
        ke[4, 3] = lm
        ke[1, 2] = mn
        ke[2, 1] = mn
        ke[4, 5] = mn
        ke[5, 4] = mn
        ke[0, 2] = ln
        ke[2, 0] = ln
        ke[3, 5] = ln
        ke[5, 3] = ln
        ke[0, 3] = -ll
        ke[3, 0] = -ll
        ke[1, 4] = -mm
        ke[4, 1] = -mm
        ke[2, 5] = -nn
        ke[5, 2] = -nn
        ke[0, 4] = -lm
        ke[4, 0] = -lm
        ke[1, 3] = -lm
        ke[3, 1] = -lm
        ke[1, 5] = -mn
        ke[5, 1] = -mn
        ke[2, 4] = -mn
        ke[4, 2] = -mn
        ke[0, 5] = -ln
        ke[5, 0] = -ln
        ke[2, 3] = -ln
        ke[3, 2] = -ln
    else:
        print('错误的维度信息')
        return
    ke = ke * e * a / length
    return ke


def rod_bee(bee, length):
    bee[0, 0] = -1 / length
    bee[0, 1] = 1 / length


def rod_ke(ke, e, a, length):
    """
    :param ke: 初始化的杆单元刚度矩阵
    :param e: 弹性模量
    :param a: 横截面积
    :param length: 杆单元的长度
    :return: 杆单元的单元刚度矩阵
    """
    ke[0, 0] = 1.0
    ke[1, 1] = 1.0
    ke[0, 1] = -1.0
    ke[1, 0] = -1.0
    ke = ke * e * a / length
    return ke


def sparse_cho_bac(kv, loads, k_diag):
    """
    稀疏矩阵Cholesky分解的前代和回代
    :param kv:
    :param loads:
    :param k_diag:
    :return:
    """
    n = np.size(k_diag, 0)
    loads[1] = loads[1] / kv[0]
    for i in range(1, n):
        ki = k_diag[i] - i - 1
        li = k_diag[i - 1] - ki
        x = loads[i + 1]
        if li != i:
            m = i
            for j in range(li, m):
                x = x - kv[ki + j] * loads[j + 1]
        loads[i + 1] = x / kv[ki + i]
    for it in range(1, n):
        i = n - it
        ki = k_diag[i] - i - 1
        x = loads[i + 1] / kv[ki + i]
        loads[i + 1] = x
        li = k_diag[i - 1] - ki
        if li != i:
            m = i
            for k in range(li, m):
                loads[k + 1] = loads[k + 1] - x * kv[ki + k]
    loads[1] = loads[1] / kv[0]


def sparse_cho_fac(kv, k_diag):
    """
    一维变带宽存储的稀疏矩阵Cholesky分解
    :param kv: 向量形式存储的稀疏矩阵
    :param k_diag: 对角线辅助向量
    """
    x = 0
    n = np.size(k_diag, 0)
    kv[0] = np.sqrt(kv[0])
    for i in range(1, n):
        ki = k_diag[i] - (i + 1)
        li = k_diag[i - 1] - ki + 1
        for j in range(li, i + 2):
            x = kv[ki + j - 1]
            kj = k_diag[j - 1] - j
            if j != 1:
                ll = k_diag[j - 2] - kj + 1
                ll = max(li, ll)
                if ll != j:
                    m = j
                    for k in range(ll, m):
                        x = x - kv[ki + k - 1] * kv[kj + k - 1]
            kv[ki + j - 1] = x / kv[kj + j - 1]
        kv[ki + i] = np.sqrt(x)