param(
    [string]$Archive = "01_Train_Val_Oil_Spill_images.7z",
    [string]$OutDir = "extracted_dataset\images",
    [string]$ListFile = "missing_files_list.txt"
)

$7zExe = "C:\Program Files\7-Zip\7z.exe"
if (-not (Test-Path $7zExe)) {
    Write-Error "7z.exe not found at $7zExe"
    exit 1
}

$missingCount = (Get-Content $ListFile | Where-Object { $_.Trim() -ne '' }).Count
Write-Host "Targeting $missingCount missing files from listfile: $ListFile"
Write-Host "Destination: $OutDir"
Write-Host "Archive: $Archive"
Write-Host "Invoking 7-Zip with -aos (skip existing files)..."

# In 7-Zip command line syntax, @listfile must not be broken by space in path.
# Using relative path within current working directory avoids spaces!
$args = @(
    "e",
    $Archive,
    "-o$OutDir",
    "-aos",
    "-y",
    "@$ListFile"
)

$process = Start-Process -FilePath $7zExe -ArgumentList $args -NoNewWindow -PassThru -Wait
Write-Host "7-Zip completed with exit code: $($process.ExitCode)"
exit $process.ExitCode
