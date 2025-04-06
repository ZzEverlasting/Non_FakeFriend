// script.js

// Function to navigate between pages
function navigateTo(page) {
    window.location.href = page;
}

// Event listener for transitioning to "app.html"
document.getElementById('App').addEventListener('click', function () {
    navigateTo('app.html');
});

// Event listener for transitioning back to "home.html"
document.getElementById('goToHome').addEventListener('click', function () {
    navigateTo('home.html');
});

// Event listener for starting the recording
document.getElementById('startRecording').addEventListener('click', function () {
    console.log("Recording started...");
    
});
// Event listener for stopping the recording
document.getElementById('stopRecording').addEventListener('click', function () {
    // Logic to stop recording
    console.log("Recording stopped.");
});
// Event listener for playing the recording
document.getElementById('playRecording').addEventListener('click', function () {
    // Logic to play the recording
    console.log("Playing recording...");
});
// Event listener for saving the recording