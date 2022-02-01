<?php
if (isset($_POST['submit'])) {
    $file = $_FILES['file'];

    $fileName = $file['name'];
    $fileTmpName = $file['tmp_name'];
    $fileSize = $file['size'];
    $fileError = $file['error'];
    $fileType = $file['type'];

    $fileExt = explode('.', $fileName);
    $fileActualExt = strtolower(end($fileExt));

    $allow = array('jpg', 'jpeg', 'png');

    if (in_array($fileActualExt, $allow)) {
        if ($fileError === 0) {
            if ($fileSize < 1000000) {
                $fileNameNew = uniqid('', true).'.'.$fileActualExt;
                $fileDestination = 'uploads/'.$fileNameNew;
                move_uploaded_file($fileTmpName, $fileDestination);

                header("Location: upload_image.html?UploadSuccess");
            } else {
                echo "Your file is too large";
            }
        } else {
            echo "There was an error uploading the file";
        }
    } else {
        echo "Wrong format file";
    }
}

?>