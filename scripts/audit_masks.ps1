param(
    [string]$ImagesDir = "extracted_dataset\images",
    [string]$MasksDir = "extracted_dataset\masks"
)

Write-Output "Auditing Mask and Image pairs..."

$imgFiles = Get-ChildItem -Path $ImagesDir -Filter "*.tif" | Sort-Object Name
$maskFiles = Get-ChildItem -Path $MasksDir -Filter "*.tif" | Sort-Object Name

Write-Output "Total Images Found: $($imgFiles.Count)"
Write-Output "Total Masks Found:  $($maskFiles.Count)"

$imgSet = New-Object 'System.Collections.Generic.HashSet[string]'
foreach ($f in $imgFiles) { [void]$imgSet.Add($f.Name) }

$maskSet = New-Object 'System.Collections.Generic.HashSet[string]'
foreach ($f in $maskFiles) { [void]$maskSet.Add($f.Name) }

$missingMasks = @()
foreach ($f in $imgFiles) {
    if (-not $maskSet.Contains($f.Name)) {
        $missingMasks += $f.Name
    }
}

$missingImages = @()
foreach ($f in $maskFiles) {
    if (-not $imgSet.Contains($f.Name)) {
        $missingImages += $f.Name
    }
}

Write-Output "Missing Masks (Images without Mask): $($missingMasks.Count)"
Write-Output "Missing Images (Masks without Image): $($missingImages.Count)"

# Inspect TIFF structure of sample masks
# Mask size is 4,211,084 bytes (2048 x 2048 = 4,194,304 bytes + 16,780 header bytes)
$sampleMasks = @($maskFiles[0], $maskFiles[10], $maskFiles[100], $maskFiles[500], $maskFiles[1000], $maskFiles[-1])

$uniqueValuesSet = New-Object 'System.Collections.Generic.HashSet[byte]'
$pixelCountPerVal = @{}

foreach ($sm in $sampleMasks) {
    $bytes = [System.IO.File]::ReadAllBytes($sm.FullName)
    # TIFF header analysis
    $isLittleEndian = ($bytes[0] -eq 0x49 -and $bytes[1] -eq 0x49)
    $isBigEndian = ($bytes[0] -eq 0x4D -and $bytes[1] -eq 0x4D)
    
    # In these TIFFs, let's scan all bytes in the raster strip
    # Mask is single-band 8-bit uncompressed uint8 (2048 x 2048)
    # The header is at the start (16780 bytes) and raster data follows, or vice versa.
    # Let's inspect unique bytes in the file
    $localSet = New-Object 'System.Collections.Generic.HashSet[byte]'
    for ($i = 0; $i -lt $bytes.Length; $i++) {
        $b = $bytes[$i]
        [void]$localSet.Add($b)
        [void]$uniqueValuesSet.Add($b)
    }
    Write-Output "File: $($sm.Name), Size: $($bytes.Length) bytes, Endian: $(if($isBigEndian){'BigEndian'}else{'LittleEndian'}), UniqueByteCount: $($localSet.Count)"
}

# Scan exact raster pixel data for first 20 masks to accurately measure oil vs background values
Write-Output "Scanning exact raster pixel distribution across 20 representative masks..."
$oilPixelsTotal = 0
$bgPixelsTotal = 0
$otherPixelsTotal = 0
$oilMaskCount = 0

# Tag parsing for uncompressed mask strip offset
foreach ($sm in ($maskFiles | Select-Object -First 20)) {
    $stream = [System.IO.File]::OpenRead($sm.FullName)
    $header = New-Object byte[] 4096
    $stream.Read($header, 0, 4096) | Out-Null
    
    # BigEndian TIFF: 4D 4D 00 2A
    $firstIfd = ([uint32]$header[4] -shl 24) -bor ([uint32]$header[5] -shl 16) -bor ([uint32]$header[6] -shl 8) -bor [uint32]$header[7]
    $numEntries = ([uint16]$header[$firstIfd] -shl 8) -bor [uint16]$header[$firstIfd+1]
    
    $stripOffset = 0
    $stripByteCounts = 0
    $width = 0
    $height = 0
    $bitsPerSample = 0
    
    for ($i = 0; $i -lt $numEntries; $i++) {
        $entryOffset = $firstIfd + 2 + ($i * 12)
        $tag = ([uint16]$header[$entryOffset] -shl 8) -bor [uint16]$header[$entryOffset+1]
        $valOffset = $entryOffset + 8
        if ($tag -eq 256) { $width = ([uint16]$header[$valOffset] -shl 8) -bor [uint16]$header[$valOffset+1] }
        if ($tag -eq 257) { $height = ([uint16]$header[$valOffset] -shl 8) -bor [uint16]$header[$valOffset+1] }
        if ($tag -eq 258) { $bitsPerSample = ([uint16]$header[$valOffset] -shl 8) -bor [uint16]$header[$valOffset+1] }
        if ($tag -eq 273) { 
            # StripOffsets
            $stripOffset = ([uint32]$header[$valOffset] -shl 24) -bor ([uint32]$header[$valOffset+1] -shl 16) -bor ([uint32]$header[$valOffset+2] -shl 8) -bor [uint32]$header[$valOffset+3] 
        }
        if ($tag -eq 279) { 
            # StripByteCounts
            $stripByteCounts = ([uint32]$header[$valOffset] -shl 24) -bor ([uint32]$header[$valOffset+1] -shl 16) -bor ([uint32]$header[$valOffset+2] -shl 8) -bor [uint32]$header[$valOffset+3] 
        }
    }
    
    # Read raster pixels
    $raster = New-Object byte[] $stripByteCounts
    $stream.Seek($stripOffset, [System.IO.SeekOrigin]::Begin) | Out-Null
    $stream.Read($raster, 0, $stripByteCounts) | Out-Null
    $stream.Close()
    
    $localOil = 0
    $localBg = 0
    $localOther = 0
    $localVals = New-Object 'System.Collections.Generic.HashSet[byte]'
    
    for ($p = 0; $p -lt $raster.Length; $p++) {
        $val = $raster[$p]
        [void]$localVals.Add($val)
        if ($val -eq 0) {
            $localBg++
        } elseif ($val -eq 255) {
            $localOil++
        } else {
            $localOther++
        }
    }
    
    if ($localOil -gt 0) { $oilMaskCount++ }
    $oilPixelsTotal += $localOil
    $bgPixelsTotal += $localBg
    $otherPixelsTotal += $localOther
    
    Write-Output "Mask $($sm.Name): Dim=($width x $height), BitsPerSample=$bitsPerSample, StripOffset=$stripOffset, Values=[$($localVals -join ', ')], OilPixels=$localOil, BgPixels=$localBg, OtherPixels=$localOther"
}

Write-Output "--- SUMMARY ---"
Write-Output "Sample Oil Masks: $oilMaskCount / 20"
Write-Output "Total Sampled Oil Pixels: $oilPixelsTotal"
Write-Output "Total Sampled Bg Pixels:  $bgPixelsTotal"
Write-Output "Total Sampled Other Pixels: $otherPixelsTotal"
