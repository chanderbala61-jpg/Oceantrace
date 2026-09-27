$names = @("00000.tif", "00002.tif", "00642.tif")

foreach ($name in $names) {
    $filePath = "temp_audit_samples\images\$name"
    $bytes = [System.IO.File]::ReadAllBytes($filePath)
    
    # Read First IFD
    $firstIfd = ([uint32]$bytes[4] -shl 24) -bor ([uint32]$bytes[5] -shl 16) -bor ([uint32]$bytes[6] -shl 8) -bor [uint32]$bytes[7]
    $numEntries = ([uint16]$bytes[$firstIfd] -shl 8) -bor [uint16]$bytes[$firstIfd+1]
    
    $tileOffsetsPtr = 0
    $tileBytesPtr = 0
    $compression = 0
    
    for ($i = 0; $i -lt $numEntries; $i++) {
        $entryOffset = $firstIfd + 2 + ($i * 12)
        $tag = ([uint16]$bytes[$entryOffset] -shl 8) -bor [uint16]$bytes[$entryOffset+1]
        $valOffset = $entryOffset + 8
        if ($tag -eq 259) { $compression = ([uint16]$bytes[$valOffset] -shl 8) -bor [uint16]$bytes[$valOffset+1] }
        if ($tag -eq 324) { 
            $tileOffsetsPtr = ([uint32]$bytes[$valOffset] -shl 24) -bor ([uint32]$bytes[$valOffset+1] -shl 16) -bor ([uint32]$bytes[$valOffset+2] -shl 8) -bor [uint32]$bytes[$valOffset+3] 
        }
        if ($tag -eq 325) { 
            $tileBytesPtr = ([uint32]$bytes[$valOffset] -shl 24) -bor ([uint32]$bytes[$valOffset+1] -shl 16) -bor ([uint32]$bytes[$valOffset+2] -shl 8) -bor [uint32]$bytes[$valOffset+3] 
        }
    }
    
    # First tile offset
    $t0_off = ([uint32]$bytes[$tileOffsetsPtr] -shl 24) -bor ([uint32]$bytes[$tileOffsetsPtr+1] -shl 16) -bor ([uint32]$bytes[$tileOffsetsPtr+2] -shl 8) -bor [uint32]$bytes[$tileOffsetsPtr+3]
    $t0_len = ([uint32]$bytes[$tileBytesPtr] -shl 24) -bor ([uint32]$bytes[$tileBytesPtr+1] -shl 16) -bor ([uint32]$bytes[$tileBytesPtr+2] -shl 8) -bor [uint32]$bytes[$tileBytesPtr+3]
    
    Write-Output "File: $name | Compression: $compression (5=LZW) | Tile0: Offset $t0_off, ByteCount $t0_len"
}
