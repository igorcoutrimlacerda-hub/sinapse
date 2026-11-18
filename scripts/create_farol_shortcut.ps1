$ErrorActionPreference = "Stop"

$Repo = "C:\Users\igorc\Downloads\mem"
$Launcher = Join-Path $Repo "scripts\start_farol.ps1"
$PowerShell = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
$Shell = New-Object -ComObject WScript.Shell
$Desktop = $Shell.SpecialFolders.Item("Desktop")
$ShortcutPath = Join-Path $Desktop ("Farol " + [char]0x2014 + " Copiloto de Projetos.lnk")

$Shortcut = $Shell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = $PowerShell
$Shortcut.Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$Launcher`""
$Shortcut.WorkingDirectory = $Repo
$Shortcut.Description = "Farol " + [char]0x2014 + " Copiloto de Projetos"
$Shortcut.IconLocation = "$PowerShell,0"
$Shortcut.Save()

Get-Item $ShortcutPath | Select-Object FullName, Length, LastWriteTime
