import sys, os, time, datetime, pandas, numpy, pickle, base64, logging
from utils import fcts

table_width_common   = 315
table_style_common   = 'font-size: 72%;margin-left: auto;margin-right: auto;'
caption_style_common = 'background-color:#90805f; color:white'
table_style          = 'background-color: #F7F5F1;'
table_td_hover_style = 'table td:hover {background-color: #ebe6dd;}'


# JS CODE TO ENABLE DATATABLE MODE
datatable_code = {}
datatable_code['head'] = """<script type="text/javascript" src="https://code.jquery.com/jquery-3.5.1.js"></script>
<script type="text/javascript" src="https://cdn.datatables.net/1.10.22/js/jquery.dataTables.min.js"></script>
<link rel="stylesheet" href="http://cdn.datatables.net/1.10.22/css/jquery.dataTables.min.css">
<script>
$(document).ready( function () {
    $('#myTable').DataTable({"pageLength": 50, "aaSorting": []});
} );
</script>"""
datatable_code['table'] = """ id="myTable" """


def get_link(text, url, title='', decorate=False, add_style='', new_tab=False):
    """Returns a <a href> html link"""
    if title != '': title = ' title="' + str(title) + '"'
    decorate_str = '' if decorate else ' style="text-decoration: none; ' + add_style + '"'
    new_tab = ' target="_blank"' if new_tab else ''
    return '<a href="' + url + '" ' + title + decorate_str + new_tab + '>' + str(text) + '</a>'

def get_color(value, ref_value):
    """This function should not be here. But it is just a green/red html color selector, based on a reference value"""
    if value > ref_value * 1.05:
        return 'color:#3D9970'
    if value < ref_value * 0.95:
        return 'color:#FF4136'
    return 'color:#111111'

def img_src_base64(img_base64, hover_text=''):
    """Return a <img> markup, with the image as a base64 serialized text. img_base64 should be an image object."""
    if hover_text != '':
        hover_text = ' title="' + str(hover_text) + '"'
    
    if img_base64 != '':
        return '<img ' + hover_text + ' src="data:image/jpeg;base64,' + img_base64.decode() +'" ></img>'
    else:
        return ''

def wrap_image_base64(img_path, trailing_br=False, **kwargs):
    """Wrap base64 image from a file. Return a <img> markup."""
    _width = kwargs.get('width', '')
    _style = kwargs.get('style', '')
    
    if _style != '':
        _style = ' style="' + _style + '"'
    
    if str(_width) != '' : _width = ' width='+str(_width)
    else                 : _width = ''
    
    img_base64 = open(img_path, 'rb').read()    # read bytes from file
    img_base64 = base64.b64encode(img_base64)   # encode to base64 (bytes)
    img_base64 = img_base64.decode()            # convert bytes to string
    
    txt = '<img src="data:image/jpeg;base64,' + img_base64 +'" ' + _width + _style + '></img>'
    if trailing_br:
        txt = txt + '<br>'
    return txt

def wrap_image(img_path, trailing_br=False, **kwargs):
    """Return a <img> markup, linking to the image in a URL"""
    _width = kwargs.get('width', '')
    _style = kwargs.get('style', '')
    
    if _style != '':
        _style = ' style="' + _style + '"'
    
    if str(_width) != '' : _width = ' width='+str(_width)
    else                 : _width = ''
    
    txt = '<img src="' + img_path +'" ' + _width + _style + '></img>'
    if trailing_br:
        txt = txt + '<br>'
    return txt

def add_html_hoover(text, hoover_text, text_color='', url=''):
    """Add a "title" to a text, using an empty href link, in order to create a mouseover"""
    if text_color != '': text_color = '; color: ' + text_color
    return '<a href="'+url+'" style="text-decoration: none' + text_color + '" title="' + hoover_text + '">' + text + '</a>'


class HtmlDoc():
    doc_close_code = '</body></html>\n'
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.init_doc(kwargs)
        self.closed = False
        self.out_path = kwargs.get('out_path', '')
        self.check_out_path_exists()
        self.img_mode = kwargs.get('img_mode', 'file')
    
    def __str__(self):
        return 'HtmlDoc. File: ' + str(self.out_path)
    
    def check_out_path_exists(self):
        """Make sure the output directory exists. Otherwise, create it"""
        if self.out_path != '':
            fcts.create_missing_dir(os.path.dirname(self.out_path))
    
    def add_default_style(self, kwargs):
        body_style     = kwargs.get('body_style', 'body{ max-width: 650px;margin:0 auto; }')
        self.open_code += """<style>
                    table, th, td {
                      border-collapse: collapse;
                      border: 0px solid black;
                      text-align: """ + kwargs.get('text_align', 'center') + """;
                      padding: 4px;
                      border-bottom: 1px solid #ddd;
                      """ + kwargs.get('additional_table_style', '') + """
                    }
                    
                    """ + kwargs.get('td_hover_style', '') + """
                    br {display: block; /* makes it have a width */
                    content: ""; /* clears default height */
                    margin-top: 5px; /* change this to whatever height you want it */}
                    
                    @media print{.page-break  { display:block; page-break-before:always; }}
                    
                    
                    .tooltip {
                      position: relative;
                      display: inline-block;
                    }

                    .tooltip .tooltiptext {
                      visibility: hidden;
                      width: 400px;
                      background-color: black;
                      color: #fff;
                      text-align: left;
                      border-radius: 6px;
                      padding: 5px 0;
                      padding-left: 10px 0;
                      
                      /* Position the tooltip */
                      position: absolute;
                      z-index: 1;
                      top: 100%;
                      left: 50%;
                      margin-left: -60px;
                    }

                    .tooltip:hover .tooltiptext {
                      visibility: visible;
                    }
                    
                    
                    """ + body_style + """

                    </style>"""
    
    def page_break(self, **kwargs):
        """Insert a pagebreak, to make sure that a printed document will go to the next page at this point"""
        page_break_code = '<div class="page-break"></div>'
        
        if kwargs.get('position', 'bottom') == 'bottom':
            self.code += page_break_code
        else:
            self.code_top += page_break_code
    
    def init_doc(self, kwargs):
        """Initialize the HTML document"""
        title     = kwargs.get('title', '')
        body_font = kwargs.get('body_font', 'font-family:calibri;')
        
        self.code = ''
        
        # Placeholder, to add some code at the top of the page
        self.code_top = ''
        
        self.open_code = '<!DOCTYPE html><html><head><title>' + title + '</title></head><body style="' + body_font + '">\n'
        if kwargs.get('include_datatable', False):
            self.open_code = self.open_code + datatable_code['head']
        
        self.add_default_style(kwargs)
    
    def close_doc(self):
        """Close the HTML document"""
        self.code = self.open_code + self.code_top + self.code + self.doc_close_code
        self.closed = True
    
    def br(self, n=1, nbsp=False, **kwargs):
        """Add <br> markup to the document"""
        for i in range(n):
            code = '<br>'
            if nbsp:
                code += '&nbsp;'
            
            if kwargs.get('position', 'bottom') == 'bottom':
                self.code += code
            else:
                self.code_top += code
    
    def hr(self, **kwargs):
        """Add <hr> markup (line separator) to the document"""
        default_style = 'width:60%; border: 0;height: 2px;background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0));'
        style = kwargs.get('style', default_style)
        self.code += '<hr style="' + style + '"/>\n'
    
    def text(self, text, txt_type='p', style='', *args, **kwargs):
        """Add a line of text to the document"""
        if kwargs.get('position', 'bottom') == 'bottom':
            self.code += '<' + txt_type+ ' style="' + str(style) + '">' + text + '</' + txt_type + '>\n'
        else:
            self.code_top += '<' + txt_type+ ' style="' + str(style) + '">' + text + '</' + txt_type + '>\n'
    
    def image(self, img_path, **kwargs):
        """Add img src image to the document"""
        if self.img_mode == 'embed' or kwargs.get('img_mode', 'file') == 'embed':
            code = wrap_image_base64(img_path, **kwargs)
        else:
            code = wrap_image(img_path, **kwargs)
        
        if kwargs.get('position', 'bottom') == 'bottom':
            self.code += code
        else:
            self.code_top += code
    
    def anchor(self, anchor_name):
        """Add an anchor to the document"""
        self.code += '\n<div id="' + anchor_name + '"/>'
    
    def open_float(self, side):
        """to put 2 things next to each other"""
        self.code += "<div style='float: " + side + "'>"
    
    def close_float(self):
        """Close floating section"""
        self.close_div()
    
    def close_div(self):
        """Close div section"""
        self.code += "</div>"
    
    def open_table(self, headers, code, *args, **kwargs):
        """Open <table> element"""
        style         = kwargs.get('style', '')
        th_width_arg  = kwargs.get('th_width', '') # if STR: same for all columns. If LIST: one style for each column
        table_width   = kwargs.get('table_width', '')
        headers_style = kwargs.get('headers_style', '')
        table_id      = kwargs.get('table_id', '')
        show_thead    = kwargs.get('show_thead', True)
        caption_txt   = kwargs.get('caption_txt', '')
        caption_style = kwargs.get('caption_style', '')
        
        if table_id != '':
            table_id = ' id="' + str(table_id) + '"'
        
        if table_width != '':
            table_width = ' width=' + str(table_width)
        
        if headers_style != '':
            headers_style = ' style="' + headers_style + '"'
        
        if self.kwargs.get('include_datatable', False):
            _datatable_id = datatable_code['table']
        else:
            _datatable_id = ''
        
        code +=  '\n<table style="' + style + '" ' + table_width + ' ' + _datatable_id + ' ' + table_id + '>\n'
        
        if caption_txt != '':
            caption_code = '<caption style="' + caption_style + '">' + caption_txt + '</caption>'
            code += caption_code
        
        if show_thead:
            code += '<thead><tr>'
            for i,x in enumerate(headers):
                if isinstance(th_width_arg, list):
                    th_width = th_width_arg[i]
                else:
                    th_width = th_width_arg
                
                code += '<th ' + th_width + ' ' + headers_style + '>' + str(x) + '</th>\n'
            code += '</tr></thead>'
        
        return code
    
    def add_row(self, row, code, **kwargs):
        """Add row (<tr>) to the <table> element"""
        code += '<tr style="line-height: 13px">'
        for i, cell_txt in enumerate(row):
            
            # Vertical separator, not for the last cell (right border)
            if kwargs.get('vertical_separator', '') != '':
                add_style_txt = 'border-right: ' + kwargs['vertical_separator']
                
                if isinstance(cell_txt, list):
                    cell_txt[1] = cell_txt[1] + ';' + add_style_txt
                else:
                    cell_txt = [cell_txt, add_style_txt]
            
            code = self.add_cell(cell_txt, code)
        code += '</tr>\n'
        return code
    
    def add_cell(self, cell_txt, code):
        """Add cell (<td>) to the <tr> element"""
        style_str = ''
        
        if isinstance(cell_txt, list):
            if len(cell_txt) > 1:
                style_str = 'style="' + str(cell_txt[1]) + '"'
            
            cell_txt = cell_txt[0]
        
        code += '<td ' + style_str + '>' + str(cell_txt) + '</td>'
        return code
    
    def add_table_contents(self, rows_list, code, **kwargs):
        """Add all rows to table"""
        for row in rows_list:
            code = self.add_row(row, code, **kwargs)
        return code
    
    def add_df(self, df, *args, **kwargs):
        """Convert a DF and add it to the document"""
        table = [list(df.columns)] + list(df.values)
        self.add_table(table, *args, **kwargs)
    
    def add_table(self, table, *args, **kwargs):
        """Add a 2D list to the document"""
        code = ''
        code = self.open_table(table[0], code, *args, **kwargs)
        code = self.add_table_contents(table[1:], code, **kwargs)
        code = self.close_table(code)
        
        if kwargs.get('position', 'bottom') == 'bottom':
            self.code += code
        else:
            self.code_top += code
        
        # Unless specified otherwise, add a BR after the table
        if kwargs.get('trailing_br', True):
            self.br()
    
    def close_table(self, code):
        """Close the <table> element"""
        return code + '\n</table>'
    
    def get(self):
        """Get the code of the html document"""
        if self.closed:
            return self.code_top + self.code
        else:
            return self.open_code + self.code_top + self.code + self.doc_close_code
    
    def fix_fpath(self, fpath):
        """Remove special characters"""
        for c in ['\n', '\t', '*', '?', '<', '>', '|']:
            fpath = fpath.replace(c, '')
        return fpath
    
    def write(self, out_path=''):
        """Write html document to a file, and return the path"""
        if out_path == '':
            if self.out_path != '':
                out_path = self.out_path
        
        if out_path != '':
            with open(self.fix_fpath(out_path), 'w',  encoding='utf-16') as fout:
                fout.writelines(self.get())
            logging.info('Wrote: ' + str(out_path))
            return out_path



# Simple table, not full document
def get_simple_table(df, style=''):
    """Generate code for a simple HTML table, without the whole document around it"""
    code  = '\n<table>'
    
    def one_row(code, items_list, bar_sep_every_n_cols=-1):
        code += '\n<tr>'
        for item in items_list:
            code += '\n<td style="' + style + '">'
            code += str(item)
            code += '\n</td>'
        code += '\n</tr>'
        return code
    
    # HEADERS (AS PART OF THE TABLE)
    code = one_row(code, df.columns)
    
    # CONTENTS
    for i,row in df.iterrows():
        code = one_row(code, [row[x] for x in df.columns])
    
    code += '\n</table>'
    return code





