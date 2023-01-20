import matplotlib as mpl


def set_notebook():
    mpl.rcParams["figure.figsize"] = (4, 4)

    
def opts(plot_kwargs: dict):
    
    def decorated(f):
        def wrapped(*args, **kwargs):
            with plt.rc_context(plot_kwargs):
                return f(*args, **kwargs)
        return wrapped
    
    return decorated