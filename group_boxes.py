import pandas as pd
import numpy as np

def rotate_mids(mid_x, mid_y, original_width, original_height, rotation):
    #print("width: {}, height: {}, orig width: {}, orig height: {}, rotation: {}".format(mid_x, mid_y, original_width, original_height, rotation))
    # rot 0 doesn't get rotated back - obviously ;)
    rot_midx = -1.0
    rot_midy = -1.0
    #print(rotation)
    if rotation == 90:
        # Translate up and rotate
        rot_midy = mid_x
        rot_midx = -1 * (mid_y - original_height)
        pass
    elif rotation == 180:
        # Translate up and left and rotate
        rot_midx = -1 * (mid_x - original_width)
        rot_midy = -1 * (mid_y - original_height)
    elif rotation == 270:
        # Translate left and rotate
        rot_midx = mid_y
        rot_midy = -1 * (mid_x - original_width)
    elif rotation == 0:
        # copy data over
        rot_midx = mid_x
        rot_midy = mid_y

    return [rot_midx, rot_midy]

def make_box_id(books):
    x = 0
    ids = []
    current = books[0]
    for i in books:
        if current != i:
            x = 0
        ids.append(x)
        current = i
        x+=1

    return ids

# Please note: this function does not work currently
def get_included_rectangle(subframe):
    # The subframe is a pandas dataframe with only images and data for the same bookspine
    subframe.set_index('text_box_id', inplace=True)
    for i in range(0, len(subframe)):
        # for each row, see if any of the others lie within this
        # calculate... 0 <= dot(AB,AM) <= dot(AB,AB) && 0 <= dot(BC,BM) <= dot(BC,BC)
        # get AB, dot(AB,AB), BC, dot(BC,BC) first
        # AB = (Bx-Ax , By-Ay) -> for us, this is (rtx-ltx, rty-lty)
        # BC = (Cx-Bx , Cy-By) -> for us, this is (rbx-rtx, rby-rty)
        curr_row = subframe.loc[subframe.index[i]]
        vec_AB = [curr_row.get('rtx') - curr_row.get('ltx'), curr_row.get('rty') - curr_row.get('lty')]
        vec_BC = [curr_row.get('rbx') - curr_row.get('rtx'), curr_row.get('rby') - curr_row.get('rty')]
        vec_AB_dot = np.dot(vec_AB, vec_AB)
        vec_BC_dot = np.dot(vec_BC, vec_BC)

        # Now iterate over all other images - this pains me
        for j in range(0, len(subframe)):
            if(i==j):
                continue

            curr_compare = subframe.loc[subframe.index[j]]
            # I can't finish this until I have the corner point coordinates in the same coordinate system - which i *don't* have!!
            vec_AM = [0.0 - curr_row.get('ltx'), 0.0 - curr_row.get('lty')]
            vec_BM = [0.0, 0.0]
            vec_ABAM_dot = [0.0, 0.0]
            vec_BCBM_dot = [0.0, 0.0]

def get_included_circle(subframe, includes):
    # The subframe is a pandas dataframe with only images and data for the same bookspine
    subframe.set_index('text_box_id', inplace=True)
    for i in range(0, len(subframe)):
        # I'll put all included IDs in this
        included = '['

        # for each row, see if any of the others lie within this
        curr_row = subframe.loc[subframe.index[i]]
        vec_AB = [curr_row.get('rtx') - curr_row.get('ltx'), curr_row.get('rty') - curr_row.get('lty')]

        mid = [curr_row.get('midx_original'), curr_row.get('midy_original')]

        w = curr_row.get('width')
        h = curr_row.get('height')
        radius = 0.0
        if w >= h:
            radius = h / 2.0
        else:
            radius = w / 2.0

        # Now iterate over all other images - this pains me
        for j in range(0, len(subframe)):
            if i == j:
                continue

            curr_compare = subframe.loc[subframe.index[j]]
            distance = np.linalg.norm(np.subtract([curr_compare.get('midx_original'),  curr_compare.get('midy_original')], mid))

            # Check if distance is within radius
            if radius-distance >= 0:
                included = included + str(j) + ','
                pass

        included = included + ']'
        # This adds a string per image - or row - in the original dataframe
        includes.append(included)


# TODO finish this function so that the output is the same as with get_included_circle
# This one is faster and essentially gets the same information, but it's not symmetrical
# So this needs to be changed before it can really be used for the pipeline
# NOTE: I sort of abandoned this since speed-up really isn't substantial enough
def get_included_circle_fast(subframe, includes):
    # The subframe is a pandas dataframe with only images and data for the same bookspine
    subframe.set_index('text_box_id', inplace=True)
    for i in range(0, len(subframe)):
        # I'll put all included IDs in this
        included = '['

        # for each row, see if any of the others lie within this
        curr_row = subframe.loc[subframe.index[i]]
        vec_AB = [curr_row.get('rtx') - curr_row.get('ltx'), curr_row.get('rty') - curr_row.get('lty')]

        mid = [curr_row.get('midx_original'), curr_row.get('midy_original')]

        w = curr_row.get('width')
        h = curr_row.get('height')
        radius = 0.0
        if w >= h:
            radius = h / 2.0
        else:
            radius = w / 2.0

        # Now iterate over all other images - this pains me
        if not ((i+1) >= len(subframe)):
            for j in range(i+1, len(subframe)):
                curr_compare = subframe.loc[subframe.index[j]]
                distance = np.linalg.norm(np.subtract([curr_compare.get('midx_original'),  curr_compare.get('midy_original')], mid))

                # Check if distance is within radius
                if radius-distance >= 0:
                    included = included + str(j) + ','

        included = included + ']'
        # This adds a string per image - or row - in the original dataframe
        includes.append(included)


def main(directory):
    data = pd.read_csv(directory + (r'/results/data.csv'))

    # calculate midpoints in original rotations (0)
    mids = [rotate_mids(mx, my, ow, oh, rot) for mx, my, ow, oh, rot in zip(data['m1'], data['m2'], data['original_width'], data['original_height'], data['rotation'])]
    data = pd.concat([data, pd.DataFrame(mids, columns=('midx_original','midy_original'))], axis=1)
    ids = make_box_id(data['book'].to_numpy())
    data = pd.concat([data, pd.DataFrame(ids, columns=('text_box_id',))], axis=1)

    inc = []
    for shelf in data['shelf'].unique():
        sub_shelf = data[data['shelf'] == shelf]
        for book in data['book'].unique():
            sub_book = sub_shelf[sub_shelf['book'] == book]
            get_included_circle(sub_book, inc)
            #get_included_circle_fast(sub_book, inc)

    data = pd.concat([data, pd.DataFrame(inc, columns=('includes',))], axis=1)
    data.to_csv(directory + (r'/results/data.csv'), index=None)
    # Group input data by book first, then feed that to some algorithm



if __name__ == '__main__':
    main()
