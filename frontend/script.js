// script.js

// Function to navigate between pages
function navigateTo(page) {
    window.location.href = page;
}

// Event listener for transitioning to "app.html"
document.getElementById('goToApp').addEventListener('click', function () {
    navigateTo('app.html');
});

// Event listener for transitioning back to "home.html"
document.getElementById('goToHome').addEventListener('click', function () {
    navigateTo('home.html');
});
