param(
    [string]$Archive = "c:\Users\bala\OneDrive\Documents\exp 1\01_Train_Val_Oil_Spill_images.7z",
    [string]$MissingCsv = "missing_images.csv",
    [string]$ListFile = "missing_files_list.txt"
)

$data = Import-Csv -Path $MissingCsv
$lines = @($data | ForEach-Object { $_.ArchivePath })
[System.IO.File]::WriteAllLines((Join-Path (Get-Location) $ListFile), $lines)

Write-Host "Wrote $($lines.Count) archive paths to $ListFile"
Write-Host "Sample entries:"
$lines[0..2] | ForEach-Object { Write-Host "  $_" }
