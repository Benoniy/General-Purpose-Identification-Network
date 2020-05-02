import wx


class WindowClass(wx.Frame):
    def __init__(self, parent, title):
        super(WindowClass, self).__init__(parent, title=title, size=(500, 600))
        self.Show()


app = wx.App()
WindowClass(None, title="GPIN")
app.MainLoop()
