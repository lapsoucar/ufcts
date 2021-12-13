import sys, os, time, datetime, pandas, numpy, pickle, logging, py7zr, base64, io, random
from functools import wraps
from matplotlib import pyplot as plt


def process_figure(out_path, plt):
    """Show / save / serrialize plot"""
    if out_path == 'base64':
        plt.tight_layout()
        out_path = serialize_image(plt)
    elif out_path != '':
        plt.tight_layout()
        plt.savefig(out_path)
    elif out_path == '':
        plt.show()
    
    return out_path

def text_to_fname(text, suffix_date=False):
    """Remove special characters, spaces, etc to make a filename"""
    text = text.replace(' ', '_')
    text = ''.join(ch for ch in text if ch.isalnum())
    
    if suffix_date:
        text += datetime.datetime.now().strftime('%Y%m%d')
    
    return text

def to_zipped_protected_csv(df, out_path, password):
    """Save dataframe to an encrypted 7zip file"""
    try:
        df.to_csv(out_path, index=None,sep=";")
        zip_path = out_path+'.zip'
        with py7zr.SevenZipFile(zip_path, 'w', password=password) as zf:
            zf.writeall(out_path)
        time.sleep(0.01)
        os.remove(out_path)
        return zip_path
    except Exception as err:
        logging.info('Error writing to_zipped_protected_csv (' + str(out_path) + '): ' + str(err))
        return ''

def timed(func):
    """This decorator prints the execution time for the decorated function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logging.info("{fct_name} ran in {time_s}s".format(fct_name=func.__name__, time_s=round(end - start, 2)))
        return result

    return wrapper

def get_trendpoly(xs,ys):
    """Polynomial regression of a series"""
    trend = numpy.polyfit(xs,ys,1)
    trendpoly = numpy.poly1d(trend) 
    return trendpoly

def show_trendpoly(ax, xs, ys, color='red'):
    """Add polynomial regression to a plot"""
    trendpoly = get_trendpoly(xs, ys)
    X = [xs.min(), xs.max()]
    Y = trendpoly(X)
    ax.plot(X,Y, color=color)
    return trendpoly

def is_any_in_txt(txt_list, within_txt):
    """
        Within (txt_list), is there one item contained in within_txt ?
        Example: (['a', 'b'], 'ab') --> Yes, a is contained in ab
    """
    for x in txt_list:
        if x in within_txt:
            return True
    return False

def is_in_any_txt(txt, within_txts_list, case_insensitive=False):
    """is "txt" in any of the texts list ?"""
    for within_txt in within_txts_list:
        
        if case_insensitive: # slower
            if txt.lower() in within_txt.lower():
                return True
        else:
            if txt in within_txt:
                return True
    return False

def try_div(x,y, def_value=0):
    """Divide x/y, and return the default value in case of error (typically if we divide by 0"""
    try:
        return x/y
    except:
        return def_value

def list_unique(x):
    """Remove duplicates from a list, without losing order of items"""
    _ = []
    for i in x:
        if not i in _:
            _.append(i)
    return _

def showcount(df, col_name, topn=0, print_count=True, as_percentage=False):
    """Group by a column and show how many items in each group"""
    len_tot     = len(df)
    
    if isinstance(col_name, str):
        n_distinct = len(df[col_name].unique())
    else:
        n_distinct = len(df[col_name].drop_duplicates())
    
    if print_count:
        print(str(n_distinct) + ' distinct values')
    
    if isinstance(col_name, str):
        tmp_df      = df[[col_name]].copy()
        tmp_df['n'] = tmp_df.groupby(col_name)[col_name].transform(len)
        tmp_df['%'] = tmp_df['n'] / len_tot
        tmp_df      = tmp_df[[col_name, 'n', '%']].drop_duplicates().sort_values('n', ascending=False)
    else:
        tmp_df      = df[col_name].copy()
        tmp_df['n'] = tmp_df.groupby(col_name)[col_name[0]].transform(len)
        tmp_df['%'] = tmp_df['n'] / len_tot
        tmp_df      = tmp_df[col_name + ['n', '%']].drop_duplicates().sort_values('n', ascending=False)
    
    if as_percentage:
        tmp_df['%'] = tmp_df['%'].map(lambda x: to_pc(100*x))
    
    if topn == 0:
        return tmp_df
    else:
        return tmp_df.head(topn)

def mem_usage(pandas_obj, ret_type='str'):
    """Show usage of a pandas object"""
    if isinstance(pandas_obj,pandas.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else: # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    if ret_type == 'str':
        return "{:03.2f} MB".format(usage_mb)
    else:
        return usage_mb

def try_to_category(df, col_name):
    """Convert to category if possible"""
    try:
        df[col_name] = df[col_name].astype('category')
    except:
        pass
    return df

def categorize_all(all_tx, cat_cols):
    """Convert select columns types to categories (save some memory...)"""
    for c in cat_cols:
        try:
            size_bf = mem_usage(all_tx[c])
            all_tx[c] = all_tx[c].astype('category')
            size_af = mem_usage(all_tx[c])
            logging.info('categorize_all: ' + c + '. Size : ' + str(size_bf) + ' --> ' + str(size_af))
        except Exception as err:
            logging.info('Error categorizing: ' + str(c) + ': ' + str(err))
    return all_tx

def to_pickle(obj, fpath):
    """Save object to a pickle file"""
    with open(fpath, "wb") as f:
        pickle.dump(obj, f)

def read_pickle(fpath):
    """Read object from a pickle file"""
    with open(fpath, "rb") as f:
        return pickle.load(f)

def sign(x):
    """Get the sign of a number"""
    return 1 if x >=0 else -1

def all_same_signs(x_list):
    """Returns True if all items in the list have the same sign"""
    return len(set([sign(x) for x in x_list])) == 1

def log(msg, logmode='print'):
    """Write to logfile, with a timestamp"""
    msg_full = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f ' + str(msg))
    if logmode == 'print':
        print(msg_full)
    else:
        logging.info(str(msg))

def f_thousands_k(n, sep="'"):
    """Format number to thousands, with a k replacing 000"""
    if not isinstance(n, int):
        if isinstance(n, float) and numpy.isnan(n):
            return 'None'
        else:
            n = int(n)
    
    if n > 999:
        n = int(round(n/1000))
        return f"{n:,d}".replace(',', sep) + 'k'
    else:
        return f"{n:,d}".replace(',', sep)

def f_thousands(n, sep="'"):
    """Format number to thousands"""
    if not isinstance(n, int):
        if isinstance(n, float) and numpy.isnan(n):
            return 'None'
        else:
            n = int(n)
    
    return f"{n:,d}".replace(',', sep)

def f_big(n, sep="'"):
    """Format big number"""
    if n > 10*1000*1000:
        n = round(n / 1000000, 0)
        suffix = 'M'
        format_code = '{:.0f}'
    elif n > 1*1000*1000:
        n = round(n / 1000000, 1)
        suffix = 'M'
        format_code = '{:.1f}'
    elif n > 10*1000:
        n = round(n / 1000, 0)
        suffix = 'k'
        format_code = '{:.0f}'
    elif n > 1000:
        n = round(n / 1000, 1)
        suffix = 'k'
        format_code = '{:.1f}'
    else:
        n = round(n, 0)
        suffix = ''
        format_code = '{:.0f}'
    
    return format_code.format(n) + suffix

def create_missing_dir(dir_path):
    """Create a directory if it doesnt exist"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path

def replace_all_str(x, pattern_list, repl_by):
    """Replace all patterns by a str"""
    for p in pattern_list:
        x = x.replace(p, repl_by)
    return x

def replace_str_all_empty(x, repl_by=''):
    """Replace #, NA, N/A, etc..."""
    x = x.strip()
    l = ['#', '?', 'n//a', 'n/a', 'not defined', 'unknown', 'nan']
    for ll in l:
        if x == ll:
            x = repl_by
    return x

def get_dates_range_str(df, col_name='date'):
    """Format a range of dates to a string yyyymmdd --> yyyymmdd"""
    return df[col_name].min().strftime('%Y-%m-%d -> ') + df[col_name].max().strftime('%Y-%m-%d')

def todf(table):
    """2D list to dataframe. The first row is the headers"""
    return pandas.DataFrame(table[1:], columns=table[0])

def clip(df):
    """Clip DF to clipboard, without index"""
    df.to_clipboard(index=False)

def fdict(d, col_like, col_like_2='', show_values=False):
    """Search for a key containing col_like"""
    if show_values:
        return [(k,v) for k,v in d.items() if col_like.lower() in k.lower() and col_like_2.lower() in k.lower()]
    else:
        return [k for k in d.keys() if col_like.lower() in k.lower() and col_like_2.lower() in k.lower()]

def fcol(df, col_like, col_like_2='', **kwargs):
    """Find col name in df"""
    if isinstance(df, dict):
        return fdict(df, col_like, col_like_2, **kwargs)
    else:
        nrows = kwargs.get('nrows', 2)
        return df[[c for c in df if col_like in c.lower() and col_like_2.lower() in c.lower()]].drop_duplicates().head(nrows)

def tryround(x, *args):
    """Try rounding a number. In case of failure, return 0"""
    try: return round(x, *args)
    except:
        return 0

def try_strftime(x, *args, **kwargs):
    """Try strftime. In case of failure, return an empty string"""
    try:
        return x.strftime(*args, **kwargs)
    except:
        return ''

def try_to_datetime(x, frmt=''):
    """Try to convert a string to a date. In case of failure, return nan"""
    try:
        if frmt == '':
            return pandas.to_datetime(x)
        else:
            return pandas.to_datetime(x, format=frmt)
    except:
        return numpy.nan

def float_to_datetime(x):
    """20190102.0 --> remove decimals and parse to datetime"""
    try:
        x = str(int(x))
        return try_to_datetime(x)
    except:
        return numpy.nan

def capitalize_all(x):
    """Capitalize all words of a sentence"""
    _str = [word.capitalize() for word in x.split(' ')]
    return ' '.join(_str)

def clean_email(x):
    """Simple cleaning on email addresses"""
    x = x.replace(' ', '')
    x = x.replace('<', '')
    x = x.replace('>', '')
    x = x.replace("'", "")
    x = x.replace('"', '')
    return x

def describe_column(c, df, i):
    """Describe a DF in a column"""
    col_desc = [i
                , str(df[c].dtype)
                , str(c).replace('\n','')
                , ', '.join(map(str, df[c].unique()[:3]))[:20]+'...']
    return col_desc

def describe(df):
    """Describe a DF"""
    df_desc = [['Col Nr', 'Type', 'Name', 'Examples']]
    for i,c in enumerate(df):
        df_desc.append(describe_column(c, df, i))
    return todf(df_desc)

def showcols(df):
    """Show available columns names for a DF"""
    cols = list(df.columns)
    for i in range(10):
        if len(cols[i*5:(i+1)*5])>0:
            print(', '.join(cols[i*5:(i+1)*5]))

def min_nonan(xseries):
    """Minimum value of a series, ignoring nans"""
    try:
        return min(xseries[-xseries.isnull()])
    except:
        return numpy.nan

def max_nonan(xseries):
    """Maximum value of a series, ignoring nans"""
    try:
        return max(xseries[-xseries.isnull()])
    except:
        return numpy.nan

def getfig(x=3,y=3):
    """Clear Matplotlib memory and generate a new figure"""
    plt.clf()
    plt.close('all')
    plt.figure(figsize=(x,y))

def table_dates(df, date_col, ym='ym', agg_col='count'): # agg_col: if this is a column name, then sum it
    """Simple table, counting entries by date group"""
    df_copy = df[[date_col]].copy() if agg_col == 'count' else df[[date_col, agg_col]].copy()
    if ym == 'ym':
        df_copy['date'] = df[date_col].map(lambda x: datetime.datetime.strptime(x.strftime('%Y-%m-01'), '%Y-%m-%d'))
    else:
        df_copy['date'] = df[date_col].map(lambda x: datetime.datetime.strptime(x.strftime('%Y-01-01'), '%Y-%m-%d'))
    
    if agg_col == 'count':
        df_copy['n'] = df_copy.groupby('date')['date'].transform(len)
    else:
        df_copy['n'] = df_copy.groupby('date')[agg_col].transform(sum)
    
    df_copy = df_copy[['date', 'n']].drop_duplicates().sort_values('date')
    return df_copy

def plot_dates(df, date_col, ym='ym', agg_col='count'):
    """Simple plot X = date, Y = Nb entries"""
    df_copy = table_dates(df, date_col, ym=ym, agg_col=agg_col)
    plt.plot(df_copy['date'], df_copy['n'])
    plt.grid(None)
    return

def join2(xseries, sep=',', final_sep=' or '):
    """Regular "join", but with a special separator before the last item. Example: join2([a,b,c,d], ',', ' and ') --> 'a, b, c and d' """
    return sep.join(xseries[:-1]) + final_sep + ''.join(xseries[-1:])

def prplus(number):
    """Prefix with "+" for positive values"""
    return '+' if number >= 0 else ''

def to_pc(n, force_decimals=99, add_plus=False, brackets=False):
    """Format as a percentage"""
    n_original = n
    if force_decimals != 99:
        n = round(n, force_decimals)
    elif  n >= 0.6: n = int(round(n,0))
    else: n = round(n,1)
    
    # "-0.0" is not changing to 0 by itself
    if n == -0.0: n = 0.0
    
    n_rounded = n
    
    n = str(n)+'%'
    if n.endswith('.0%'): n = n.replace('.0%', '%')
    if add_plus: n = prplus(n_rounded) + n
    if brackets: n = '(' + n + ')'
    return n

def stround(x, force_dec=-99):
    """automatic str(round(x))"""
    if force_dec != -99:
        return str(round(x, force_dec))
    #else
    if x < 0.05:  force_dec = 4
    elif x < 0.5: force_dec = 3
    elif x < 2:   force_dec = 2
    else:         force_dec = 0
    return str(round(x, force_dec))

def nest(d: dict) -> dict:
    result = {}
    for key, value in d.items():
        target = result
        for k in key[:-1]:  # traverse all keys but the last
            target = target.setdefault(k, {})
        target[key[-1]] = value
    return result

def df_to_nested_dict(df: pandas.DataFrame) -> dict:
    d = df.to_dict(orient='index')
    return {k: nest(v) for k, v in d.items()}

def serialize_image(fig_obj):
    """Image to base64 serialized"""
    pic_IObytes = io.BytesIO()
    fig_obj.savefig(pic_IObytes,  format='png')
    pic_IObytes.seek(0)
    pic_hash = base64.b64encode(pic_IObytes.read())
    return pic_hash

def renorm_series(series, new_min, new_max):
    """Normalize series in a given range"""
    old_min = series.min()
    old_max = series.max()
    delta1 = old_max - old_min
    delta2 = new_max - new_min
    series = series.map(lambda n: new_min + (delta2 * (n - old_min) / delta1))
    return series

def click(x,y):
    """Mouseclick at a certain location"""
    import win32api, win32con
    win32api.SetCursorPos((x,y))
