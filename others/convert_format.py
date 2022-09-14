def split_data_into_fts_lbls(data_fname, data_fts_fname, data_lbls_fname):
    """
    Split the training data into features and labels files.
    """
    inpfile = open(data_fname, 'r')
    ftfile = open(data_fts_fname, 'w')
    lblfile = open(data_lbls_fname, 'w')

    ctr = 0
    num_valid = 0
    for line in inpfile:
        line = line.rstrip()
        if ctr != 0:
            items = line.split(None, 1)
            if not line.startswith(' ') and len(items) == 2:
                lbls = items[0].split(',')
                lblfile.write(' '.join([lbl + ':1' for lbl in lbls]) + '\n')
                ftfile.write(items[1] + '\n')
                num_valid += 1
            else:
                lblfile.write('\n')
                ftfile.write(items[0] + '\n')
        else:
            items = line.split()
            num_inst = items[0]
            num_ft = items[1]
            num_lbl = items[2]

            ftfile.write(num_inst + ' ' + num_ft + '\n')
            lblfile.write(num_inst + ' ' + num_lbl + '\n')

        ctr += 1

    inpfile.close()
    ftfile.close()
    lblfile.close()

    num_inst = str(num_valid)
    # No need to check if num_ft and num_lbl is None as it is the first line.
    with open(data_fts_fname, 'r') as f:
        lines = f.readlines()
        lines[0] = num_inst + ' ' + num_ft + '\n'
    with open(data_fts_fname, 'w') as f:
        f.writelines(lines)

    with open(data_lbls_fname, 'r') as f:
        lines = f.readlines()
        lines[0] = num_inst + ' ' + num_lbl + '\n'
    with open(data_lbls_fname, 'w') as f:
        f.writelines(lines)
