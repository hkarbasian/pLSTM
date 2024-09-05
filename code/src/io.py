import h5py

def export_hdf5(fname, X, dname):
    h5f = h5py.File('{}'.format(fname), 'w')
    h5f.create_dataset('{}'.format(dname), data=X)
    h5f.close()
    return

def import_hdf5(fname, dname):
    h5f = h5py.File(fname, 'r')
    data = h5f['{}'.format(dname)]
    return data