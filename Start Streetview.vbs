Option Explicit
Dim files, shell, workspace, python, launcher
Set files = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")
workspace = files.GetParentFolderName(WScript.ScriptFullName)
python = files.BuildPath(workspace, ".venv\Scripts\pythonw.exe")
launcher = files.BuildPath(workspace, "tools\streetview_app\launch.py")
If Not files.FileExists(python) Then
  MsgBox "Create .venv and install this project first. See README.md.", 48, "Streetview to PLY"
  WScript.Quit 1
End If
shell.CurrentDirectory = workspace
shell.Run """" & python & """ -B """ & launcher & """", 0, False

