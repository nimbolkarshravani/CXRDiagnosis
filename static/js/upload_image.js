$(document).ready(function () {
    $('.image-section').hide();
    $('#predictbtn').hide();
});
function readURL(input) {
    $('.image-section').show();
    $('#predictbtn').show();

    if (input.files && input.files[0]) {
        var reader = new FileReader();

        reader.onload = function (e) {
            $('#uploadedImage')
                .attr('src', e.target.result)
                .width(350)
                .height(350);
        };

        reader.readAsDataURL(input.files[0]);
    }
}