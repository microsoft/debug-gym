window.HELP_IMPROVE_VIDEOJS = false;


$(document).ready(function() {
    // Check for click events on the navbar burger icon

    var options = {
			slidesToScroll: 1,
			slidesToShow: 1,
			loop: true,
			infinite: true,
			autoplay: true,
			autoplaySpeed: 5000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);
	
    bulmaSlider.attach();

})

  function onConsentChanged() {
	// function gtag() {
	//   window.dataLayer.push(arguments)
	// }
  
	// if (!consentRequired() || WcpConsent.siteConsent.getConsentFor(WcpConsent.consentCategories.Analytics)) {
	//   // Load GA
	//   loadAnalytics(gtag)
	// }
  }

  function consentRequired() {
	return WcpConsent.siteConsent.isConsentRequired;
  }
  
  $(function() {
	// Load GA upfront because we classify it as essential cookie
	window.dataLayer = window.dataLayer || []
	function gtag() {
	  dataLayer.push(arguments)
	}
	gtag('js', new Date())
  
	window.WcpConsent && WcpConsent.init("en-US", "cookie-banner", function (err, _siteConsent) {
	}, onConsentChanged, WcpConsent.themes.light);
  
	// const cookieManager = document.querySelector('#footer-cookie-link');
	// if (consentRequired() && cookieManager && cookieManager.parentElement) {
	//   cookieManager.parentElement.style.visibility = 'visible';
	// }
  
	// initialize consent
	onConsentChanged();
  })
