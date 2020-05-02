import wx
import os

NAME = ""
SIZE = 0
EPOCHS = 0
BATCH_SIZE = 0
data_path = ""


class WindowClass(wx.Frame):
    def __init__(self, *args, **kwargs):
        super(WindowClass, self).__init__(*args, **kwargs)
        self.create_gui()

    def create_gui(self):
        menu_bar = wx.MenuBar()
        file_button = wx.Menu()
        open_item = file_button.Append(wx.ID_OPEN, "Open...", "Status msg...")
        save_item = file_button.Append(wx.ID_SAVE, "Save...", "Status msg...")
        exit_item = file_button.Append(wx.ID_EXIT, "Exit", "Status msg...")

        menu_bar.Append(file_button, "&File")
        self.SetMenuBar(menu_bar)

        self.Bind(wx.EVT_MENU, self.quit, exit_item)
        self.Bind(wx.EVT_MENU, self.load_cfg, open_item)
        self.Bind(wx.EVT_MENU, self.save_cfg, save_item)

        self.SetTitle("GPIN")
        self.Show(True)

    def save_cfg(self, e):
        file_dialogue = wx.FileDialog(self, "Save", "", "", "File type (*.cfg)|*.cfg", wx.FD_SAVE)
        file_dialogue.ShowModal()
        file_path = file_dialogue.GetPath()
        file_dialogue.Destroy()
        save_config(file_path)

    def load_cfg(self, e):
        file_dialogue = wx.FileDialog(self, "Open a file", "", "", "File type (*.cfg)|*.cfg", wx.FD_OPEN)
        file_dialogue.ShowModal()
        file_path = file_dialogue.GetPath()
        file_dialogue.Destroy()
        read_config(file_path)

    def quit(self, e):
        self.Close()


def main():
    app = wx.App()
    WindowClass(None)
    app.MainLoop()


def read_config(file_name):
    global NAME, SIZE, EPOCHS, BATCH_SIZE, data_path
    if ".cfg" not in file_name.lower():
        file_name = file_name + ".cfg"

    if os.path.exists(file_name):
        file = open(file_name, "r")
        f = file.read()
        list = f.split("\n")
        for t in list:
            if "name" in t.lower():
                NAME = t.split("=")[1]
            elif "size" in t.lower():
                SIZE = int(t.split("=")[1])
            elif "epochs" in t.lower():
                EPOCHS = int(t.split("=")[1])
            elif "batch" in t.lower():
                BATCH_SIZE = int(t.split("=")[1])
            elif "path" in t.lower():
                data_path = t.split("=")[1]

        file.close()
        return True
    return False


# Creates a save file
def save_config(file_name):
    global NAME, SIZE, EPOCHS, BATCH_SIZE, data_path
    if ".cfg" not in file_name.lower():
        file_name = file_name + ".cfg"

    if not os.path.exists(file_name):
        file = open(file_name, "w")
        file.write("NAME=" + str(NAME) +
                   "\nSIZE=" + str(SIZE) +
                   "\nEPOCHS=" + str(EPOCHS) +
                   "\nBATCH=" + str(BATCH_SIZE) +
                   "\nPATH=" + str(data_path))
        file.close()
        return True
    return False


main()
