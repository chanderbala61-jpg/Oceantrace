$filePath = "temp_audit_samples\images\00000.tif"
$bytes = [System.IO.File]::ReadAllBytes($filePath)

# Search XML for Band information
$xmlStart = 568
$xmlLen = 5659076
$sampleChunk = [System.Text.Encoding]::UTF8.GetString($bytes, $xmlStart, 20000)

Write-Output "=== BAND NAMES & POLARIZATION IN XML ==="
$lines = $sampleChunk -split "`n"
foreach ($line in $lines) {
    if ($line -like "*BAND_NAME*" -or $line -like "*POLARISATION*" -or $line -like "*polarisation*" -or $line -like "*BAND_INDEX*" -or $line -like "*DATA_TYPE*" -or $line -like "*unit*") {
        Write-Output $line.Trim()
    }
}

Write-Output "`n=== TILE OFFSETS AND BYTE COUNTS ==="
# Tag 324 is TileOffsets (count 16 at 248), Tag 325 is TileByteCounts (count 16 at 312)
for ($t=0; $t -lt 4; $t++) {
    $off = 248 + ($t * 4)
    $bOff = [System.BitConverter]::ToUInt32($bytes, $off)
    if ([System.BitConverter]::IsLittleEndian) {
        $sub = $bytes[$off..($off+3)]
        [Array]::Reverse($sub)
        $tileOffset = [System.BitConverter]::ToUInt32($sub, 0)
    }
    
    $cOff = 312 + ($t * 4)
    $subC = $bytes[$cOff..($cOff+3)]
    if ([System.BitConverter]::IsLittleEndian) { [Array]::Reverse($subC) }
    $tileCount = [System.BitConverter]::ToUInt32($subC, 0)
    
    Write-Output "Tile $t: Offset $tileOffset, ByteCount $tileCount"
}

# Check raw pixel values from first uncompressed/compressed tile
# Tag 259 was 5 (LZW compression).
