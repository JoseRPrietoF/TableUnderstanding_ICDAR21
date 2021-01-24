import glob, os, copy, pickle
from xml.dom import minidom
import numpy as np
import matplotlib.pyplot as plt
import cv2
# print(cv2.__version__)
from scipy.signal import convolve2d
import shapely
from shapely.geometry import LineString, Point
# construct the Laplacian kernel used to detect edge-like
# regions of an image


# construct the Sobel x-axis kernel
horizontal_kernel = np.array((
    [-1, -1, -1],
    [2, 2, 2],
    [-1, -1, -1]), dtype="int")
# horizontal_kernel = np.array((
#     [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
#     [ 2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
#     [1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), dtype="int")

# construct the Sobel y-axis kernel
vertical_kernel = np.array((
    [-1, 2, -1],
    [-1, 2, -1],
    [-1, 2, -1]), dtype="int")


THICKNESS = 10
THRESHOLD = 100 #nº pixels
BASELINES = True
BINARY = False
DPI = 600
TWO_DIM = False
SHOW_IMG = False
COLS = True
ROWS = True
TABLE_BOX = True

iGRID_STEP = 33 # odd number is better

class TablePAGE():
    """
    Class for parse Tables from PAGE
    """

    def __init__(self, im_path, debug=False,
                 search_on=["TextLine"]):
        """
        Set filename of inf file
        example : AP-GT_Reg-LinHds-LinWrds.inf
        :param fname:
        """
        self.im_path = im_path
        self.DEBUG_ = debug
        self.search_on = search_on

        self.parse()



    def get_daddy(self, node, searching="TextRegion"):
        while node.parentNode:
            node = node.parentNode
            if node.nodeName.strip() == searching:
                return node
        return None

    def get_text(self, node, nodeName="Unicode"):
        TextEquiv = None
        for i in node.childNodes:
            if i.nodeName == 'TextEquiv':
                TextEquiv = i
                break
        if TextEquiv is None:
            print("No se ha encontrado TextEquiv en una región")
            return None

        for i in TextEquiv.childNodes:
            if i.nodeName == nodeName:
                try:
                    words = i.firstChild.nodeValue
                except:
                    words = ""
                return words

        return None

    def get_TableRegion(self, ):
        """
        Return all the cells in a PAGE
        :return: [(coords, col, row)], dict, dict
        """
        cells = []
        for region in self.xmldoc.getElementsByTagName("TableRegion"):
            coords = self.get_coords(region)
            cells.append(coords)

        return cells

    def get_cells(self, ):
        """
        Return all the cells in a PAGE
        :return: [(coords, col, row)], dict, dict
        """
        cells = []
        cell_by_row = {}
        cell_by_col= {}
        for region in self.xmldoc.getElementsByTagName("TableCell"):
            #TODO different tables
            coords = self.get_coords(region)

            row = int(region.attributes["row"].value)
            col = int(region.attributes["col"].value)
            cells.append((coords, col, row))

            cols = cell_by_col.get(col, [])
            cols.append(coords)
            cell_by_col[col] = cols

            rows = cell_by_row.get(row, [])
            rows.append(coords)
            cell_by_row[row] = rows

        return cells, cell_by_col, cell_by_row

    def get_cells_with_header(self, ):
        """
        Return all the cells in a PAGE
        :return: [(coords, col, row)], dict, dict
        """
        cells = []
        cell_by_row = {}
        cell_by_col= {}
        for region in self.xmldoc.getElementsByTagName("TableCell"):
            #TODO different tables
            coords = self.get_coords(region)
            DU_Header = "O"
            for child in region.childNodes:
                att = child.attributes
                if att and "DU_header" in list(att.keys()):
                    DU_Header = att["DU_header"].value
                    break
            row = int(region.attributes["row"].value)
            col = int(region.attributes["col"].value)
            cells.append((coords, col, row, DU_Header))

            cols = cell_by_col.get(col, [])
            cols.append(coords)
            cell_by_col[col] = cols

            rows = cell_by_row.get(row, [])
            rows.append(coords)
            cell_by_row[row] = rows

        return cells, cell_by_col, cell_by_row

    def get_Separator(self, vertical=True):
        """
        Return all the GridSeparator in a PAGE (TranskribusDU - ABTTableGrid)
        :return: [(coords, col, row)], dict, dict
        """
        cells = []
        for region in self.xmldoc.getElementsByTagName("SeparatorRegion"):
            # orient_reg = int(region.attributes["orient"])
            orient_reg = "vertical" in region.attributes["orient"].value
            # if orient == orient_reg:
            if vertical == orient_reg :
                coords = self.get_coords(region)
                cells.append(coords)
        return cells

    def get_GridSeparator(self, orient=90):
        """
        Return all the GridSeparator in a PAGE (TranskribusDU - ABTTableGrid)
        :return: [(coords, col, row)], dict, dict
        """
        cells = []
        for region in self.xmldoc.getElementsByTagName("GridSeparator"):
            orient_reg = int(region.attributes["orient"].value)
            # orient_reg = "vertical" in region.attributes["orient"].value
            if orient == orient_reg:
            # if vertical == orient_reg :
                coords = self.get_coords(region)
                cells.append(coords)
        return cells

    def get_Baselines(self, ):
        """
        A partir de un elemento del DOM devuelve, para cada textLine, sus coordenadas y su contenido
        :param dom_element:
        :return: [(coords, words)]
        """
        text_lines = []
        for region in self.xmldoc.getElementsByTagName("Baseline"):
            coords = region.attributes["points"].value
            coords = coords.split()
            coords_to_append = []
            for c in coords:
                x, y = c.split(",")
                coords_to_append.append((int(x), int(y)))

            text_lines.append(coords_to_append)


        return text_lines



    def get_textLines(self, ):
        """
        A partir de un elemento del DOM devuelve, para cada textLine, sus coordenadas y su contenido
        :param dom_element:
        :return: [(coords, words)]
        """
        text_lines = []
        for region in self.xmldoc.getElementsByTagName("TextLine"):
            coords = self.get_coords(region)
            text = self.get_text(region)

            text_lines.append((coords, text))


        return text_lines

    def get_textLinesWithObject(self, ):
        """
        A partir de un elemento del DOM devuelve, para cada textLine, sus coordenadas y su contenido
        :param dom_element:
        :return: [(coords, words)]
        """
        text_lines = []
        for region in self.xmldoc.getElementsByTagName("TextLine"):
            coords = self.get_coords(region)
            text = self.get_text(region)

            text_lines.append((coords, text, region))


        return text_lines
    
    def get_width_coords(self, coords):
        min_x = min([x[0] for x in coords])
        max_x = max([x[0] for x in coords])
        return max_x - min_x

    def coords_blank_line(self, coords, width_m=0.3, height_m = 0.1):
        min_x = min([x[0] for x in coords])
        max_x = max([x[0] for x in coords])
        min_y = min([x[1] for x in coords])
        max_y = max([x[1] for x in coords])
        width = max_x - min_x
        height = max_y - min_y
        rest_w = int((width-(width*width_m)) / 2)
        min_x += rest_w
        max_x -= rest_w
        rest_h = int((height-(height*height_m)) / 2)
        min_y += rest_h
        max_y -= rest_h
        return [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]


    def get_textLinesFromCell(self, maxWidth=9999999, empty_cells=False):
        """
        A partir de un elemento del DOM devuelve, para cada textLine, sus coordenadas y su contenido
        :param dom_element:
        :return: [(coords, words)]
        """
        text_lines = []
        for region in self.xmldoc.getElementsByTagName("TextLine"):
            id = region.attributes["id"].value
            coords = self.get_coords(region)
            if not coords:
                continue
            width_line = self.get_width_coords(coords)
            if width_line > maxWidth:
                continue
            text = self.get_text(region)
            

            tablecell = self.get_daddy(region, "TableCell")
            row, col = -1,-1
            if tablecell is not None:
                row, col = int(tablecell.attributes["row"].value), int(tablecell.attributes["col"].value)
            # print(id, row, col)
            try:
                DU_row, DU_col, DU_header = region.attributes["DU_row"].value, \
                                            region.attributes["DU_col"].value, \
                                            region.attributes["DU_header"].value
            except:
                DU_row, DU_col, DU_header = row, col, 0




            text_lines.append((coords, text, id, {
                'DU_row': DU_row,
                'DU_col': DU_col,
                'DU_header': DU_header,
                'row': row,
                'col': col,
                'fake': False
            }))
        if empty_cells:
            # Celdas vacias
            for region in self.xmldoc.getElementsByTagName("TableCell"):
                if self.has_TextLine(region):
                    continue
                coords = self.get_coords(region)
                coords = self.coords_blank_line(coords)
                
                width_line = self.get_width_coords(coords)

                id = region.attributes["id"].value

                row, col = int(region.attributes["row"].value), int(region.attributes["col"].value)
                # print(id, row, col)
                try:
                    DU_row, DU_col, DU_header = region.attributes["DU_row"].value, \
                                                region.attributes["DU_col"].value, \
                                                region.attributes["DU_header"].value
                except:
                    DU_row, DU_col, DU_header = row, col, 0

                text_lines.append((coords, "", id, {
                    'DU_row': DU_row,
                    'DU_col': DU_col,
                    'DU_header': DU_header,
                    'row': row,
                    'col': col,
                    'fake': True
                }))

        return text_lines

    def get_textLinesFromCellHisClima(self, width):
        """
        A partir de un elemento del DOM devuelve, para cada textLine, sus coordenadas y su contenido
        :param dom_element:
        :return: [(coords, words)]
        """
        max_width = width*0.4
        text_lines = []
        for region in self.xmldoc.getElementsByTagName("TextLine"):
            coords = self.get_coords(region)
            bl = np.array(coords)
            bb = max(bl[:, 0]) - min(bl[:, 0])
            if bb >= max_width:
                continue
            text = self.get_text(region)


            tablecell = self.get_daddy(region, "TableCell")
            try:
                DU_row, DU_col = region.attributes["row"].value, \
                                            region.attributes["col"].value
            except:
                DU_row = -1
                DU_col = -1



            id = region.attributes["id"].value
            row, col = -1,-1
            if tablecell is not None:
                row, col = int(tablecell.attributes["row"].value), int(tablecell.attributes["col"].value)

            text_lines.append((coords, text, id, {
                'DU_row': DU_row,
                'DU_col': DU_col,
                'row': row,
                'col': col,
            }))

        return text_lines

    def get_coords(self, dom_element):
        """
        Devuelve las coordenadas de un elemento. Coords
        :param dom_element:
        :return: ((pos), (pos2), (pos3), (pos4)) es un poligono. Sentido agujas del reloj
        """
        coords_element = None
        for i in dom_element.childNodes:
            if i.nodeName == 'Coords':
                coords_element = i
                break
        if coords_element is None:
            print("No se ha encontrado coordenadas en una región")
            return None

        coords = coords_element.attributes["points"].value
        coords = coords.split()
        coords_to_append = []
        for c in coords:
            x, y = c.split(",")
            coords_to_append.append((int(x), int(y)))
        return coords_to_append
    
    def has_TextLine(self, dom_element):
        """
        
        """
        for i in dom_element.childNodes:
            if i.nodeName == 'TextLine':
                return True
        return False


    def parse(self):
        self.xmldoc = minidom.parse(self.im_path)

    def get_width(self):
        page = self.xmldoc.getElementsByTagName('Page')[0]
        return int(page.attributes["imageWidth"].value)

    def get_height(self):
        page = self.xmldoc.getElementsByTagName('Page')[0]
        return int(page.attributes["imageHeight"].value)

def get_all_xml(path, ext="xml"):
    file_names = glob.glob("{}*.{}".format(path,ext))
    return file_names


def make_cells(cells, width, height):
    """
    make all the cells
    :param cells:
    :param width:
    :param height:
    :return:
    """
    drawing = np.zeros((height, width, 3), np.int32)
    for i in range(len(cells)):
        cell_,_,_ = cells[i]
        cell = np.array([cell_], dtype = np.int32)
        cv2.polylines(drawing, [cell],
             isClosed = True,
              color = (255,255,255),
              # color = 255,
              thickness = 10,)
    return drawing


def show_cells(drawing, title=""):
    """
    Show the image
    :return:
    """
    plt.title(title)
    plt.imshow(drawing)
    plt.show()
    plt.close()


def show_all_imgs(drawing, title="", redim=None):
    """
    Show the image
    :return:
    """
    plt.title(title)
    fig, ax = plt.subplots(1, len(drawing))
    for i in range(len(drawing)):
        if redim is not None:
            a = resize(drawing[i], redim)
            print(a.shape)
            ax[i].imshow(a)
        else:
            ax[i].imshow(drawing[i])
    plt.savefig("aqui.eps", format='eps', dpi=900)
    plt.show( dpi=900)
    plt.close()

def show_all(img, cells, cols, rows, cols_processed, rows_processed, title="", dest=None):
    """
    Show the image
    :return:
    """
    plt.title(title)
    fig, ax = plt.subplots(3, 2)
    # drawing -= drawing.min()
    # drawing /= drawing.max()
    ax[0, 0].imshow(img)
    ax[0, 1].imshow(cells)
    ax[1, 0].imshow(cols)
    ax[1, 1].imshow(rows)
    ax[2, 0].imshow(cols_processed)
    ax[2, 1].imshow(rows_processed)
    if dest is not None:
        plt.savefig(dest, format='eps', dpi=DPI)
    else:
        if SHOW_IMG:
            plt.show()
    plt.close()

def show_all_with_lines(img, cells, cols, rows, cols_processed, rows_processed, lines, title="", dest=None):
    """
    Show the image
    :return:
    """
    plt.title(title)
    fig, ax = plt.subplots(3, 3)
    # drawing -= drawing.min()
    # drawing /= drawing.max()
    ax[0, 0].imshow(img)
    ax[0, 1].imshow(cells)
    ax[0, 2].imshow(lines)
    ax[1, 0].imshow(cols)
    ax[1, 1].imshow(rows)
    ax[2, 0].imshow(cols_processed)
    ax[2, 1].imshow(rows_processed)
    if dest is not None:
        plt.savefig(dest, format='eps', dpi=DPI)
    else:
        if SHOW_IMG:
            plt.show()
    plt.close()

def clarify_horizontal(bw):
    """
    Remove pixels from and image
    :param bw:
    :return:
    """
    WINDOW_SIZE = 1 # pixels
    THRESHOLD = np.max(bw) * 500
    h_line = 0
    # First, search if there is a line
    for i in range(WINDOW_SIZE, bw.shape[0], WINDOW_SIZE):
        windowed = bw[i-WINDOW_SIZE:i, :]
        val_window = windowed.sum().sum()
        if val_window < THRESHOLD:
            h_line += 1
            bw[i - WINDOW_SIZE:i, :] = 0
    return bw

def clarify_vertical(bw):
    """
    Remove pixels from and image
    :param bw:
    :return:
    """
    WINDOW_SIZE = 1 # pixels
    THRESHOLD = np.max(bw) * 100
    h_line = 0
    # First, search if there is a line
    for i in range(WINDOW_SIZE, bw.shape[1], WINDOW_SIZE):
        windowed = bw[:, i-WINDOW_SIZE:i]
        val_window = windowed.sum().sum()
        if val_window < THRESHOLD:
            h_line += 1
            bw[:, i - WINDOW_SIZE:i] = 0
    return bw

def get_axis_from_points(coords_, n=2, axis=1, func="max"):
    """

    :param coords:
    :param n: number of points to  get
    :param axis: y=1, x=0
    :return:
    """
    points = []
    coords = copy.copy(coords_)
    if type(coords[0]) == list:
        for i in range(n):
            if func == "max":
                m = np.argmax([x[axis] for x in coords])
            else:
                m = np.argmin([x[axis] for x in coords])
            points.append(coords[m])
            # print(coords)
            # print(m)
            del coords[m]
    else:

        aux = [[x[0],x[1]] for x in coords]
        for i in range(n):
            if func == "max":
                m = np.argmax([x[axis] for x in aux])
            else:
                m = np.argmin([x[axis] for x in aux])
            points.append(aux[m])
            del aux[m]

    return points



def get_lines(tables, img, cells, cell_by_col, cell_by_row, height, width, size=None, title="", dest=None, baseLines=None):
    """
    Get the lines for every row and column. Separe horitzontal and vertical
    :param cells:
    :param cell_by_col:
    :param cell_by_row:
    :param height:
    :param width:
    :return:
    """
    print(title)
    width_img, height_img,_ = img.shape
    img_cols = np.zeros((height, width), np.int32)
    img_rows = np.zeros((height, width), np.int32)
    lines_h = np.zeros((height, width), np.int32)
    lines_w = np.zeros((height, width), np.int32)
    lines_cells = np.zeros((height, width), np.int32)
    lines_box = np.zeros((height, width), np.int32)
    # n_cols = len(list(cell_by_col.keys()))
    # n_rows = len(list(cell_by_row.keys()))
    # print("A total of {} cols and {} rows on an image of {} height and {} width".format(n_cols, n_rows,  height, width))

    # [for horizontal]
    for coords, _, _ in cells:
        points = []
        points.extend(get_axis_from_points(coords, axis=1, func="max"))
        cell = np.array([points], dtype=np.int32)
        cv2.polylines(lines_h, [cell],
                      isClosed=False,
                      color=(255, 255, 255),
                      # color = 255,
                      thickness=THICKNESS, )

        points = []
        points.extend(get_axis_from_points(coords, axis=1, func="min"))
        cell = np.array([points], dtype=np.int32)
        cv2.polylines(lines_h, [cell],
                      isClosed=True,
                      color=(255, 255, 255),
                      # color = 255,
                      thickness=THICKNESS, )

    lines_h = convolve2d(lines_h, np.ones((5,1),np.uint8), 'same')
    # [for vertical]
    # for i in list(cell_by_col.keys()):
    #     coords = cell_by_col[i]
    for coords,_,_ in cells:

        points = []
        points.extend(get_axis_from_points(coords, axis=0, func="max"))

        cell = np.array([points], dtype=np.int32)
        cv2.polylines(lines_w, [cell],
                      isClosed=False,
                      color=(255, 255, 255),
                      # color = 255,
                      thickness=THICKNESS, )
        points = []
        points.extend(get_axis_from_points(coords, axis=0, func="min"))
        cell = np.array([points], dtype=np.int32)
        cv2.polylines(lines_w, [cell],
                      isClosed=False,
                      color=(255, 255, 255),
                      # color = 255,
                      thickness=THICKNESS, )

    """CELLS"""
    box = []
    for cell,_,_ in cells:
        box.extend(cell)

        cell = np.array([cell], dtype=np.int32)

        cv2.polylines(lines_cells, [cell],
                      isClosed=True,
                      color=(255, 255, 255),
                      # color = 255,
                      thickness=THICKNESS, )
    """End cells"""


    """Table shape
    This piece of code detects the rectangular or not rectangular shape of the table :)
    """
    coco_labels = []
    for coords in tables:

        x = [p[0] for p in coords]
        y = [p[1] for p in coords]
        centroid = [(sum(x) / len(coords))/height_img, (sum(y) / len(coords))/width_img]
        min_x = min(x)
        max_x = max(x)
        min_y = min(y)
        max_y = max(y)
        width_ = (max_y - min_y) / width_img
        height_ = (max_x - min_x) / height_img
        coco_labels.append((centroid[0], centroid[1], width_, height_))
        cell = np.array([coords], dtype=np.int32)
        cv2.polylines(lines_box, [cell],
                      isClosed=True,
                      color=(255, 255, 255),
                      # color = 255,
                      thickness=THICKNESS, )
    # show_cells(lines_cells, title)
    # show_cells(img, title)

    """End table shape"""

    """textlines"""
    if baseLines is not None:
        textLinesDrawing = np.zeros((height, width), np.int32)
        for coords in baseLines:
            # coords, text = textline
            coords = np.array([coords], dtype=np.int32)
            cv2.polylines(textLinesDrawing, [coords],
                          isClosed=False,
                          color=(255, 255, 255),
                          # color = 255,
                          thickness=THICKNESS, )

    # show_cells(lines_cells, title)
    # show_cells(img, title)
    """End textlines"""
    lines_w = convolve2d(lines_w, np.ones((1,5),np.uint8), 'same')
    lines_w = np.where(lines_w > 0, 1, 0)
    lines_h = np.where(lines_h > 0, 1, 0)
    lines_h = clarify_horizontal(lines_h)
    # lines_w = clarify_vertical(lines_w)
    #Resize
    if resize is not None:
        img = resize(img, size=size)
        lines_cells = resize(lines_cells, size=size)
        lines_h = resize(lines_h, size=size)
        lines_w = resize(lines_w, size=size)
        lines_box = resize(lines_box, size=size)
        if baseLines is not None:
            textLinesDrawing = resize(textLinesDrawing, size=size)
    # show_cells(lines_h, title)
    # show_cells(lines_w, title)

    # exit()
    if not TWO_DIM:
        cols = get_1dim(lines_h, axis=1)
        rows = get_1dim(lines_w, axis=0)
    else:
        cols = get_2dim(lines_h, axis=1)
        rows = get_2dim(lines_w, axis=0)

    cols_img = create_img(cols, size[0], reverse=True)
    rows_img = create_img(rows, size[1])
    # show_cells(cols_img, title)
    # show_cells(rows_img, title)

    lines_cells = np.where(lines_cells > 127, 1, 0)

    if baseLines is not None:
        show_all_with_lines(img, lines_cells, lines_h,
                            lines_w, cols_img, rows_img, lines=textLinesDrawing,
                            title=title, dest=dest)
        # lines_cells = img_to_onehot(lines_cells, color_dict)
        return lines_cells, cols, rows, textLinesDrawing, lines_box, lines_h, lines_w, coco_labels
    else:
        show_all(resize(img, size=size), resize(lines_cells, size=size), lines_h, lines_w,
                 cols_img, rows_img, title=title, dest=dest)
        # lines_cells = img_to_onehot(lines_cells, color_dict)
        return lines_cells, cols, rows, lines_box, lines_h, lines_w, coco_labels

def get_Separators(img, page, height, width, size=None, title="", dest=None):
    """
    Get the lines for every row and column. Separe horitzontal and vertical
    :param cells:
    :param cell_by_col:
    :param cell_by_row:
    :param height:
    :param width:
    :return:
    """
    print(title)
    width_img, height_img,_ = img.shape
    img_cols = np.zeros((height, width), np.int32)
    img_rows = np.zeros((height, width), np.int32)

    # n_cols = len(list(cell_by_col.keys()))
    # n_rows = len(list(cell_by_row.keys()))
    # print("A total of {} cols and {} rows on an image of {} height and {} width".format(n_cols, n_rows,  height, width))

    cols = page.get_Separator(vertical=True)
    print(len(cols))
    # cols = page.get_GridSeparator(orient=90)
    for coords in cols:
        # coords, text = textline
        coords = np.array([coords], dtype=np.int32)
        cv2.polylines(img_cols, [coords],
                      isClosed=False,
                      color=(255, 255, 255),
                      # color = 255,
                      thickness=THICKNESS, )

    rows = page.get_Separator(vertical=False)
    # rows =  page.get_GridSeparator(orient=0)
    for coords in rows:
        # coords, text = textline
        coords = np.array([coords], dtype=np.int32)
        cv2.polylines(img_rows, [coords],
                      isClosed=False,
                      color=(255, 255, 255),
                      # color = 255,
                      thickness=THICKNESS, )


    return img_cols, img_rows, cols, rows

def create_img(cols, size, reverse=False):
    if type(cols) is not tuple:
        if not reverse:
            a =  np.array([cols]*size)
        else:
            a =  np.array([cols]*size).T
    else:
        cols_, starts = cols
        a = np.zeros((len(cols_), size))
        for i in range(len(cols_)):
            start = starts[i]
            num_pix = cols_[i]
            nums = int((num_pix*size))
            a[i, start:start+nums] = 1
        if not reverse:
            a = a.T
    return a

def get_1dim(bw, axis=0):
    """
    get the result for the split model
    :param arr:
    :return:
    """
    if BINARY:
        threshold = THRESHOLD * np.max(bw)
        bw = bw.sum(axis=axis)
        bw = np.where(bw > threshold, 1, 0)
    else:
        div = bw.shape[axis]
        bw = bw.sum(axis=axis)/div
    return bw


def get_2dim(bw, axis=0):
    """
    get the result for the split model
    :param arr:
    :return:
    """
    div = bw.shape[axis]
    sum_pix = bw.sum(axis=axis)/div
    starts = np.argmax(bw, axis)
    return (sum_pix, starts)

def resize(img, size=(1024,512)):
    return cv2.resize(img.astype('float32'), size).astype('int32')

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_to_file(data, fname):
    with open(fname, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def save_to_cocofile(data, fname):
    with open(fname, 'w') as handle:
        c = 0
        for x_center, y_center, width, height in data:
            handle.write("{} {} {} {} {}".format(c, x_center, y_center, width, height))

def load_image(path, size):
    if os.path.exists(path+".jpg"):
        p = path+".jpg"
    elif os.path.exists(path+".JPG"):
        p = path+".JPG"
    elif os.path.exists(path+".png"):
        p = path+".png"
    elif os.path.exists(path+".PNG"):
        p = path+".PNG"
    image = cv2.imread(p)
    image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = resize(image2, size=size)
    image = (image.transpose((2, 0, 1))).astype(np.float32)
    image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image, image2

color_dict = {0: 0,
              1: 1,}

def img_to_onehot(img, color_dict):
    num_classes = len(color_dict)
    shape = img.shape[:2]+(num_classes,)
    arr = np.zeros( shape, dtype=np.int8 )
    for i, cls in enumerate(color_dict):
        arr[:,:,i] = (img.flatten() == color_dict[i]).reshape(shape[:2])
    return arr

def onehot_to_img(onehot, color_dict):
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros( onehot.shape[:2] )
    for k in color_dict.keys():
        output[single_layer==k] = color_dict[k]
    return np.uint8(output)

""" From DUTranskribus"""
def getGridBB(iPageWidth, iPageHeight):
        xmin = 0
        xmax = (iPageWidth // iGRID_STEP) * iGRID_STEP
        ymin = 0
        ymax = (iPageHeight // iGRID_STEP) * iGRID_STEP
        return xmin, ymin, xmax, ymax

def iterGridHorizontalLines(iPageWidth, iPageHeight):
    """
    Coord of the horizontal lines of the grid, given the page size (pixels)
    Return iterator on the (x1,y1, x2,y2)
    """
    xmin, ymin, xmax, ymax = getGridBB(iPageWidth, iPageHeight)
    # print("grid horizontal liines: ", end=" ")
    # print(xmin, ymin, xmax, ymax)
    res = []
    for y in range(ymin, ymax+1, iGRID_STEP):
        for i in range(-70,70,70):
            res.append( [(xmin, y),(xmax, min(max(0,y+i), iPageHeight))])
    return res

def create_skewed_lines(width, height, baselines):
    xmin, ymin, xmax, ymax = getGridBB(width, height)
    img_rows = np.zeros((height, width), np.int32)
    # rows = [
    #     [(xmin, 0), (xmax, 0)],
    #     [(xmin, 0), (xmax, 30)],
    #     [(xmin, 0), (xmax, 70)],
    #     [(xmin, 0), (xmax, 500)],
    #     [(xmin, 200), (xmax, 1000)],
    #     [(xmin, 0), (xmax, 1500)],
    #     [(xmin, 0), (xmax, 2000)],
    # ]
    all_edges = []
    for coords in baseLines:
        for i in range(0,len(coords)):
            if i+1 < len(coords):
                all_edges.append([coords[i], coords[i+1]])
    # print(all_edges)
    rows = iterGridHorizontalLines(width, height)
    print("Pair of coords: {} # of BL: {}".format(len(all_edges),len(baseLines)))
    print("Rows : {}  ".format(len(rows)))
    rows_results = []
    for coords in rows:
        # coords, text = textline
        line1 = LineString(coords)
        intersections = [line1.intersection(LineString(pair)) for pair in all_edges]
        if any(intersections):
            continue
        rows_results.append(coords)
        coords = np.array([coords], dtype=np.int32)
        cv2.polylines(img_rows, [coords],
                      isClosed=False,
                      color=(255, 255, 255),
                      # color = 255,
                      thickness=THICKNESS, )
    print("rows_results : {}  ".format(len(rows_results)))

    for coords in baseLines:
        # coords, text = textline
        coords = np.array([coords], dtype=np.int32)
        cv2.polylines(img_rows, [coords],
                      isClosed=False,
                      color=(255, 255, 255),
                      # color = 255,
                      thickness=THICKNESS, )
    # show_cells(img_rows)

    return rows_results, img_rows

def get_line_min_distance(coord_line, rows):
    pass

def label_separatos(separators, gt):
    """
    0 -> False, not a true separator
    1 -> True, a separator
    :param separators:
    :param gt:
    :return:
    """
    res = []
    res_check = np.zeros(len(separators))
    for sep in separators:
        res.append([sep, 0])
    for sep in gt:
        distances = [distance(sep, x) for x in separators]
        candidate = np.argmin(distances)
        res[candidate][1] = 1
        res_check[candidate] = 1
    print("A total of {} true candidades with {} on gt".format(np.sum(res_check), len(gt)))
    return res

def distance(line1, line2):
    """
    [(xmin, 0), (xmax, 0)],
    [(xmin, 0), (xmax, 30)],
    :param line1:
    :param line2:
    :return:
    """
    # return abs(line1[0][0] - line2[0][0]) + abs(line1[0][1] - line2[0][1]) + \
    #         abs(line1[1][0] - line2[1][0]) + abs(line1[1][1] - line2[1][1])
    return abs(line1[0][1] - line2[0][1]) + \
          abs(line1[1][1] - line2[1][1])

if __name__ == "__main__":
    sizes = [
            (1024,768),
             # (512,512),
             # (768,512)
             ]
    baseLines = None
    lines = None
    cols_start = None
    rows_start = None
    for size in sizes:
        dir_img = "/data/READ_ABP_TABLE/dataset111/img/"
        dir = "/data/READ_ABP_TABLE/dataset111/img/page/"
        if COLS and ROWS:
            dir_dest = "/data/READ_ABP_TABLE/preprocess_111/rows_cols_{}_{}/".format(size[0], size[1])
        elif COLS:
            dir_dest = "/data/READ_ABP_TABLE/preprocess_111/cols_{}_{}/".format(size[0], size[1])
        elif ROWS:
            dir_dest = "/data/READ_ABP_TABLE/preprocess_111/rows_{}_{}/".format(size[0], size[1])
        create_dir(dir_dest)
        fnames = get_all_xml(dir)
        count = 0
        MAX = None
        num_cols_total = 0
        max_cols_page = 0
        for fname in fnames:
            if "doc" in fname:
                continue
            file_name = fname.split("/")[-1].split(".")[0]
            path_to_save = os.path.join(dir_dest, file_name+".pkl")
            path_to_save_work = os.path.join(dir_dest, file_name+".eps")
            page = TablePAGE(im_path=fname)
            # page = TablePAGE(im_path="/data/READ_ABP_TABLE/dataset111/b.xml")
            cells, cell_by_col, cell_by_row = page.get_cells()
            tables = page.get_TableRegion()



            if BASELINES:
                baseLines = page.get_Baselines()
            # print("A total of: {} cells in {}".format(len(cells), fname))
            # print("--"*5)
            drawing = make_cells(cells, height=page.get_height(), width=page.get_width())
            img, img_print = load_image(os.path.join(dir_img, file_name), size=size)
            result = get_lines(tables, img_print, cells, cell_by_col, cell_by_row,
                                   height=page.get_height(), width=page.get_width(),
                                   size=size, title=fname, baseLines=baseLines,
                                   # dest=path_to_save_work
            )
            if len(result) == 6:
                img_cells, cols, rows, lines_cells, lines_h, lines_w, coco_labels = result
            else:
                img_cells, cols, rows, lines, lines_cells, lines_h, lines_w, coco_labels = result

            if TWO_DIM:
                cols, cols_start = cols
                rows, rows_start = rows


            hori_lines, verti_lines, cols_gt, rows_gt = get_Separators(img, page,
            height = page.get_height(), width = page.get_width(), size = size, title = fname)

            n_cols = len(cols_gt)
            if n_cols > max_cols_page:
                max_cols_page = n_cols
            num_cols_total += n_cols
            continue

            data = {
                'img': img,
                'img_cells': img_cells,
                'rows':cols,
                'cols':rows,
                'table_shape': lines_cells,
                'cols_img': lines_w,
                'rows_img': lines_h,
            }
            if cols_start is not None:
                data['rows_start'] = cols_start
                data['cols_start'] = rows_start
            if baseLines is not None:
                data['baselines'] = lines

            rows_results, img_rows = create_skewed_lines(width = page.get_width(), height=page.get_height(), baselines=baseLines)
            # show_cells(img.transpose((1,2,0)))
            # img = np.clip(data['img_cells'] + lines, 0, 1)
            # show_cells(img_cells)
            # show_cells(img)
            # save_to_file(data, fname=path_to_save)

            labeled_sep = label_separatos(rows_results, rows_gt)
            # path_to_save = os.path.join(dir_img, file_name+".txt")
            # save_to_cocofile(coco_labels, fname=path_to_save)
            img_rows_2 = np.zeros((page.get_height(), page.get_width()), np.int32)
            for coords, gt in labeled_sep:
                if gt == 1:
                    # coords, text = textline
                    coords = np.array([coords], dtype=np.int32)
                    cv2.polylines(img_rows_2, [coords],
                                  isClosed=False,
                                  color=(255, 255, 255),
                                  # color = 255,
                                  thickness=THICKNESS, )
            show_all_imgs([img_print, verti_lines, img_rows, img_rows_2], title=fname, redim=None)

            count +=1
            print("{} of {} for Size {}_{}".format(count, len(fnames), size[0], size[1]))
            if MAX is not None and count >= MAX:
                break
            # exit()
        average_cols = num_cols_total / len(fnames)
        print("Max cols per page: {}".format(max_cols_page))
        print("average_cols per page: {}".format(average_cols))
        print("Size {}_{} done".format(size[0], size[1]))
